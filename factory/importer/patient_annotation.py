from __future__ import annotations

import logging
from typing import List

from langchain_openai import ChatOpenAI
import pandas as pd
from tqdm import tqdm

from util.config_loader import load_config_api
from util.api_client import ApiClient
from llm.utils import EmbedAPI
from llm.pydantic_model import (
    PatientNERInput,
    PatientNERResponse,
    PatientNEREntity,
    PatientNEDCandidate,
    PatientNEDOtherMention,
    PatientNEDInput,
    PatientNEDResponse,
)
from llm.tool import build_patient_ner_tool, build_patient_ned_tool


logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


def patient_annotation_factory(base_importer_cls, backend: str, config_path: str = "config.ini"):
    """
    Factory returning a concrete PatientAnnotator bound to `base_importer_cls` and `backend`.
    """

    class PatientAnnotator(base_importer_cls):
        """Importer for patient annotation data."""
        
        CYPHER_QUERY_TOPK = """
        CALL db.index.vector.queryNodes($index, $k, $qe) YIELD node, score
        RETURN node.id AS id, node.label AS label, score
        ORDER BY score DESC
        LIMIT $k
        """

        CYPHER_QUERY_ICD_CHAPTERS = """
        MATCH (c:IcdChapter)
        RETURN c.chapterName AS chapterName
        ORDER BY c.chapterName ASC
        """

        def __init__(self):
            super().__init__()
            self.backend = backend

            # Embeddings
            cfg_emb = load_config_api("embedding", path=config_path)
            self.emb_api = EmbedAPI(ApiClient(cfg_emb))

            # LLM + tools
            url_llm = load_config_api("llm", path=config_path)
            self.llm = ChatOpenAI(
                api_key="EMPTY",
                base_url=url_llm,
                model_name="google/medgemma-4b-it",
                temperature=0,
                max_tokens=24000,
                top_p=0.9,
                stop=["<end_of_turn>", "</s>", "\nUser:", "\n\nUser:"],
                frequency_penalty=0.2,
                presence_penalty=0.0,
            )
            self.patient_ner_tool = build_patient_ner_tool(self.llm)
            self.patient_ned_tool = build_patient_ned_tool(self.llm)

            # Patient NED parameters
            self.target_index_name = "icd_disease_embedding"
            self.k = 10

        ###########################
        ### NER on Patient Data ###
        ###########################

        def get_icd_chapters(self) -> List[str]:
            with self._driver.session(database=self._database) as session:
                result = session.run(self.CYPHER_QUERY_ICD_CHAPTERS)
                chapters = [r["chapterName"] for r in result]
            return chapters

        def to_patient_ner_payload(
            self,
            icd_chapters: List[str],
            patient_id: str,
            encounter_id: str,
            concat_text: str,
            narrative_text: str,
        ) -> dict:
            return {
                "icd_chapters": icd_chapters,
                "patient_id": patient_id,
                "encounter_id": encounter_id,
                "concat_text": concat_text,
                "narrative_text": narrative_text,
            }
        
        def ner_mention(self, input_data: PatientNERInput) -> PatientNEREntity:
            validated = PatientNERInput.model_validate(input_data.model_dump())
            icd_chapters = validated.icd_chapters
            patient_id = validated.patient_id
            encounter_id = validated.encounter_id
            concat_text = validated.concat_text
            narrative_text = validated.narrative_text
            payload = self.to_patient_ner_payload(
                icd_chapters,
                patient_id,
                encounter_id,
                concat_text,
                narrative_text,
            )
            tool_output = self.patient_ner_tool.invoke(payload)
            response = PatientNERResponse.model_validate(tool_output)
            return response
        
        ###########################
        ### Candidate Selection ###
        ###########################
    
        def select_candidates(self, text: str) -> List[PatientNEDCandidate]:
            """Top-K vector search for a single source text."""
            embedding = self.emb_api.embed(text)
            with self._driver.session(database=self._database) as session:
                result = session.run(
                    self.CYPHER_QUERY_TOPK,
                    index=self.target_index_name,
                    k=self.k,
                    qe=embedding,
                )
                rows = [
                    {
                        "id": r["id"],
                        "label": r["label"],
                        "score": float(r["score"]),
                    }
                for r in result
            ]
            return [PatientNEDCandidate(id=r["id"], label=r["label"], score=r["score"]) for r in rows]

        ###########################
        ### NED on Patient Data ###
        ###########################
        
        def to_patient_ned_payload(
            self,
            source_concept: PatientNEREntity,
            candidate_contexts: list[PatientNEDCandidate],
            other_mentions: list[PatientNEDOtherMention] | None = None,
        ) -> dict:
            return {
                "mention": source_concept.model_dump() if isinstance(source_concept, PatientNEREntity) else source_concept,
                "candidates": [
                    c.model_dump() if isinstance(c, PatientNEDCandidate) else c
                    for c in candidate_contexts
                ],
                "other_mentions": [
                    om.model_dump() if isinstance(om, PatientNEDOtherMention) else om
                    for om in (other_mentions or [])
                ],
        }
        
        def disambiguate_mention(self, input_data: PatientNEDInput) -> PatientNEDResponse:
            validated = PatientNEDInput.model_validate(input_data.model_dump())
            mention = validated.mention
            candidates = validated.candidates
            other_mentions = validated.other_mentions
            payload = self.to_patient_ned_payload(mention, candidates, other_mentions)
            tool_output = self.patient_ned_tool.invoke(payload)
            response = PatientNEDResponse.model_validate(tool_output)
            return response

        ##############################################
        ### Orchestration: read, filter, and write ###
        ##############################################

        def test(self, patient_data_file) -> None:
            """Test NER + NED on a sample patient data."""
            df = pd.read_csv(patient_data_file, sep="\t", dtype=str).to_dict(orient="records")[0]
            ner_input_data = PatientNERInput(
                icd_chapters=self.get_icd_chapters(),
                patient_id=df["PatientID"],
                encounter_id=df["EncounterID"],
                concat_text=df["Encounter.reasonCode"] + " | " + df["ChiefComplaint"] + " | " + df["Condition"],
                narrative_text=df["Narrative"],
            )
            ner_output = self.ner_mention(ner_input_data)
            for entity in ner_output.entities:
                candidates = self.select_candidates(text=entity.text)
                ned_input_data = PatientNEDInput(
                    mention=entity,
                    candidates=candidates,
                    other_mentions=[
                        PatientNEDOtherMention(
                            text=e.text,
                            label=e.label,
                        ) for e in ner_output.entities if e.text != entity.text
                    ],
                )
                ned_response = self.disambiguate_mention(ned_input_data)
                print("\nDisambiguation result for mention:", entity.text)
                print(ned_response)
        
        def enrich_patient_data(self, patient_data_file) -> str:
            df = pd.read_csv(patient_data_file, dtype=str)
            enriched_rows = []            
            for r in tqdm(df.itertuples(index=False, name=None), total=len(df)):
                row_dict = dict(zip(df.columns, r))
                ner_input_data = PatientNERInput(
                    icd_chapters=self.get_icd_chapters(),
                    patient_id=row_dict["PatientID"],
                    encounter_id=row_dict["EncounterID"],
                    concat_text=row_dict["Encounter.reasonCode"] + " | " + row_dict["ChiefComplaint"] + " | " + row_dict["Condition"],
                    narrative_text=row_dict["Narrative"],
                )
                ner_output = self.ner_mention(ner_input_data)
                ned_entities = []
                for entity in ner_output.entities:
                    candidates = self.select_candidates(text=entity.text)
                    ned_input_data = PatientNEDInput(
                        mention=entity,
                        candidates=candidates,
                        other_mentions=[
                            PatientNEDOtherMention(
                                text=e.text,
                                label=e.label,
                            ) for e in ner_output.entities if e.text != entity.text
                        ],
                    )
                    ned_response = self.disambiguate_mention(ned_input_data)
                    ned_entities.append(ned_response.model_dump())
                row_dict["NER_Entities"] = [e.model_dump() for e in ner_output.entities]
                row_dict["NED_Entities"] = ned_entities
                row_dict['ICD10_Codes'] = [e["icd_id"] for e in ned_entities if e["icd_id"] is not None]
                enriched_rows.append(row_dict)
            enriched_df = pd.DataFrame(enriched_rows)
            output_file = patient_data_file.replace(".csv", "_enriched.csv")
            enriched_df.to_csv(output_file, sep=",", index=False)

            return output_file
        
        def import_data(self, patient_data_file: str) -> None:
            logging.info("Enriching patient data with NER and NED...")
            output_file = self.enrich_patient_data(patient_data_file)
            logging.info(f"Enriched patient data written to: {output_file}")

    return PatientAnnotator


if __name__ == "__main__":
    from util.cli_entry import run_backend_importer

    run_backend_importer(
        patient_annotation_factory,
        description="Run Ontology Mapper.",
        file_help="No file needed for ontology mapping.",
        default_base_path="./data/",
        require_file=True,
    )
