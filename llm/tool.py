from datetime import date
import json
from typing import List, Optional

from langchain_core.tools import tool
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI

from llm.prompt import NEO4J_SCHEMA

from llm.pydantic_model import (
    OntologyMappingInput,
    PatientInfoInput,
    PatientNERInput,
    PatientNEREntity,
    PatientNEDInput,
    PatientNEDCandidate,
    PatientNEDOtherMention,
    GeneralMedicalInput,
    GeneralMedicalResponse,
    CoverageRow,
    PatientCoverageResponse,
)
from llm.chain import (
    ontology_mapping_chain,
    patient_ner_chain,
    patient_ned_chain,
    clinician_explanation_chain,
    get_patient_answer_chain,
    patient_coverage_chain,
)

from llm.pipeline import text2cypher_pipeline, enhanced_graph
from llm.pipeline_patient import get_patient_views
from llm.query_factory import rank_diseases_for_patient


def build_ontology_mapper_tool(llm):
    chain = ontology_mapping_chain(llm)

    @tool("ontology_mapper", args_schema=OntologyMappingInput)
    def ontology_mapper_tool(
        source_concept: str,
        source_context: str,
        candidate_list: str
    ):
        """Map a source medical concept to the best target candidate and return a structured result."""
        result = chain.invoke({
            "source_concept": source_concept,
            "source_context": source_context,
            "candidate_list": candidate_list,
        })
        return result.model_dump()

    return ontology_mapper_tool


def build_patient_ner_tool(llm):
    chain = patient_ner_chain(llm)

    @tool("patient_ner", args_schema=PatientNERInput)
    def patient_ner_tool(
        icd_chapters: List[str],
        patient_id: str,
        encounter_id: str,
        concat_text: str,
        narrative_text: str
    ):
        """Annotate clinical text with ICD Chapter labels and return a structured result."""
        result = chain.invoke({
            "icd_chapters": icd_chapters,
            "patient_id": patient_id,
            "encounter_id": encounter_id,
            "concat_text": concat_text,
            "narrative_text": narrative_text,
        })
        return result.model_dump()

    return patient_ner_tool


def build_patient_ned_tool(llm):
    chain = patient_ned_chain(llm)

    @tool("patient_ned", args_schema=PatientNEDInput)
    def patient_ned_tool(
        mention: PatientNEREntity,
        candidates: list[PatientNEDCandidate],
        other_mentions: list[PatientNEDOtherMention] = [],
    ):
        """Disambiguate a medical mention to the best ICD code candidate and return a structured result."""
        result = chain.invoke({
            "mention": mention,
            "candidates": candidates,
            "other_mentions": other_mentions,
        })
        return result.model_dump()

    return patient_ned_tool


def build_general_medical_tool(llm: ChatOpenAI, debug: bool = False):
    explanation_chain = clinician_explanation_chain(llm)

    @tool("general_medical_executor", args_schema=GeneralMedicalInput)
    def general_medical_executor(
        question: str,
        top_k: int = 20,
    ):
        """
        Run a general medical query against the Neo4j knowledge graph, converting
        the question to Cypher, truncating results to top_k, and returning rows
        along with an LLM-generated explanation.
        """
        cypher, rows = text2cypher_pipeline(llm, question, debug=debug)

        if isinstance(rows, list) and top_k is not None:
            rows = rows[: int(top_k)]

        # Prepare JSON for the prompt
        rows_json = json.dumps(rows, default=str)

        try:
            explanation = explanation_chain.invoke(
                {
                    "schema": NEO4J_SCHEMA.strip(),
                    "question": question,
                    "cypher": cypher,
                    "rows_json": rows_json,
                }
            )
        except Exception as e:
            explanation = f"Explanation could not be generated: {e}"

        return GeneralMedicalResponse(
            cypher=cypher,
            rows=rows,
            steps=["text2cypher", "validated", "executed", "explained"],
            explanation=explanation,
        ).model_dump()

    return general_medical_executor


import json
import re
from typing import Optional
from langchain_core.tools import tool

def build_patient_info_tool(llm: ChatOpenAI):
    explain_chain = get_patient_answer_chain(llm)

    @tool("patient_info", args_schema=PatientInfoInput)
    def patient_info_tool(
        patient_id: str,
        question: str,
        encounter_date: Optional[str] = None,
    ):
        """
        Explain a patient's virtualized Neo4j 'Patient' node in clinician-friendly
        language.

        encounter_date:
          - None       -> include all encounters, UNLESS a date or 'latest'
                          can be inferred from the question.
          - "latest"   -> use only the most recent encounter.
          - "YYYY-MM-DD" -> use only encounters from that date.
        """
        # ---- 🔍 Auto-detect encounter_date from the question if not provided ----
        if encounter_date is None:
            q_lower = question.lower()

            # Detect "latest" / "most recent" etc.
            if "latest" in q_lower or "most recent" in q_lower:
                encounter_date = "latest"
            else:
                # Simple ISO date pattern, e.g. 2024-01-22
                m = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", question)
                if m:
                    encounter_date = m.group(1)

        # Now pass the possibly-inferred encounter_date into the graph layer
        patient_views = get_patient_views(
            patient_id=patient_id,
            encounter_date=encounter_date,
        )

        if not patient_views:
            patient_json = "[]"
            has_data = False
        else:
            patient_json = json.dumps(patient_views, ensure_ascii=False, indent=2)
            has_data = True

        answer = explain_chain.invoke(
            {
                "question": question,
                "patient_json": patient_json,
            }
        )

        return {
            "patient_id": patient_id,
            "question": question,
            "encounter_date": encounter_date,
            "has_data": has_data,
            "answer": answer,
            "raw_patient_view": patient_views,
        }

    return patient_info_tool


def build_patient_coverage_tool(llm: ChatOpenAI):
    chain = patient_coverage_chain(llm)

    @tool("patient_coverage")
    def patient_coverage(
        patient_id: str,
        limit: int = 20,
    ):
        """
        Deterministic single-patient coverage using the canonical Python pipeline:
        get_patient_icd_codes -> map_icd_to_hpo -> rollup_hpo_to_ancestors -> compute_coverage
        """
        # Still run the LLM chain for traceability / audit logs
        _ = chain.invoke({"patient_id": patient_id, "limit": int(limit)})

        # Use the new implementation instead of stitched Cypher
        rows = rank_diseases_for_patient(patient_id=patient_id, limit=int(limit))

        # Cast rows into CoverageRow when possible
        cast_rows: List[CoverageRow] = []
        for r in rows or []:
            try:
                cast_rows.append(CoverageRow(**r))
            except Exception:
                # If casting fails, just skip and fall back to raw dicts
                pass

        response_rows = cast_rows or rows

        return PatientCoverageResponse(
            cypher=(
                "MULTI-STEP PIPELINE: "
                "get_patient_icd_codes -> map_icd_to_hpo -> "
                "rollup_hpo_to_ancestors -> compute_coverage"
            ),
            rows=response_rows,
            steps=[
                "patient_icd",
                "icd_to_hpo",
                "rollup_ancestors",
                "coverage",
            ],
        ).model_dump()

    return patient_coverage

