import re
from typing import List, Tuple, Any, Callable, Optional
from langchain_openai import ChatOpenAI
from neo4j.exceptions import CypherSyntaxError
from langchain_neo4j import Neo4jGraph
from langchain_neo4j.chains.graph_qa.cypher_utils import CypherQueryCorrector, Schema

from llm.chain import (
    text2cypher_chain,
    validate_cypher_chain,
    diagnose_cypher_chain,
    correct_cypher_chain,
)
from llm.prompt import NEO4J_SCHEMA
from util.config_loader import load_neo4j_config

_neo4j = load_neo4j_config()
enhanced_graph = Neo4jGraph(
    url=_neo4j["url"],
    username=_neo4j["username"],
    password=_neo4j["password"],
    database=_neo4j["database"],
    enhanced_schema=True,
)

_relationships = enhanced_graph.structured_schema.get("relationships") or []
corrector_schema = [Schema(el["start"], el["type"], el["end"]) for el in _relationships]
cypher_query_corrector = CypherQueryCorrector(corrector_schema)

_CODE_FENCE_PATTERN = re.compile(
    r"```(?:cypher)?\s*(.*?)```", re.DOTALL | re.IGNORECASE
)


def strip_code_fences(text: Any) -> str:
    if not isinstance(text, str):
        return str(text)
    match = _CODE_FENCE_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def try_explain(query: str) -> Tuple[bool, Optional[str]]:
    try:
        enhanced_graph.query(f"EXPLAIN {query}")
        return True, None
    except CypherSyntaxError as e:
        return False, e.message


def text2cypher_pipeline(
    llm: ChatOpenAI,
    question: str,
    debug: bool = False,
    debug_fn: Optional[Callable[[str], None]] = None,
) -> Tuple[str, List[dict]]:
    schema = NEO4J_SCHEMA.strip()

    if debug and debug_fn is None:
        debug_fn = print

    def log(msg: str) -> None:
        if debug and debug_fn is not None:
            debug_fn(msg)

    def no_answer(reason: str) -> Tuple[str, List[dict]]:
        log(f"[NO ANSWER] {reason}")
        # You can change the first element if you want a different sentinel
        return "/* NO_ANSWER */", []

    log("=== text2cypher_pipeline: START ===")
    log(f"Question: {question!r}")

    try:
        # 1) Generate Cypher
        generated = text2cypher_chain(llm).invoke(
            {"question": question, "schema": schema}
        )
        generated = strip_code_fences(generated)
        log("Step 1: Generated Cypher:")
        log(generated)

        errors: List[str] = []

        # 2) EXPLAIN generated query
        ok, syntax_err = try_explain(generated)
        if ok:
            log("Step 2: EXPLAIN on generated query: OK")
        else:
            log("Step 2: EXPLAIN on generated query: FAILED")
            log(f"Syntax error: {syntax_err}")
            errors.append(syntax_err)

        # 3) Relationship correction
        corrected = cypher_query_corrector(generated) or generated
        corrected = strip_code_fences(corrected)
        if corrected != generated:
            log("Step 3: Relationship correction applied:")
            log(corrected)
            ok_corr, syntax_err_corr = try_explain(corrected)
            if not ok_corr:
                log("EXPLAIN on relationship-corrected query: FAILED")
                log(f"Syntax error: {syntax_err_corr}")
                errors.append(syntax_err_corr)
        else:
            log("Step 3: Relationship correction: no change")

        # 4) LLM validation
        llm_output = validate_cypher_chain(llm).invoke(
            {"question": question, "schema": schema, "cypher": corrected}
        )
        llm_errors = getattr(llm_output, "errors", None) or []

        if llm_errors:
            log("Step 4: LLM validation found errors:")
            error_messages = list({e.message for e in llm_errors})
            errors.extend(error_messages)
            for err in error_messages:
                log(f"- {err}")
        else:
            log("Step 4: LLM validation: no errors found")

        # 5) Diagnose + correct (only if we have errors)
        if errors:
            log("Step 5: Running diagnose_cypher_chain due to errors")
            diagnosis = diagnose_cypher_chain(llm).invoke(
                {
                    "question": question,
                    "schema": schema,
                    "cypher": corrected,
                    "errors": errors,
                }
            )

            if diagnosis.issues:
                log("Diagnosis issues:")
                for issue in diagnosis.issues:
                    log(f"- {issue}")
            if diagnosis.suggestions:
                log("Diagnosis suggestions:")
                for sug in diagnosis.suggestions:
                    log(f"- {sug}")

            candidate_cypher = diagnosis.fixed_cypher or corrected
            correction_errors = errors + diagnosis.issues + diagnosis.suggestions

            log("Step 5: Running correct_cypher_chain with enriched errors")
            candidate_cypher = correct_cypher_chain(llm).invoke(
                {
                    "question": question,
                    "schema": schema,
                    "cypher": candidate_cypher,
                    "errors": correction_errors,
                }
            )
            corrected = strip_code_fences(candidate_cypher)

            ok_final, syntax_err_final = try_explain(corrected)
            if not ok_final:
                log("EXPLAIN on corrected query: FAILED")
                log(f"Syntax error: {syntax_err_final}")
                # 🔴 Fallback instead of raising
                return no_answer(
                    f"Failed to produce valid Cypher after diagnosis+correction: {syntax_err_final}"
                )
            else:
                log("EXPLAIN on corrected query: OK")
        else:
            log("Step 5: No correction needed")
            ok_final, syntax_err_final = try_explain(corrected)
            if not ok_final:
                log("Final EXPLAIN failed after 'no correction needed' path.")
                log(f"Final syntax error: {syntax_err_final}")
                # 🔴 Fallback instead of raising
                return no_answer(f"Final EXPLAIN failed: {syntax_err_final}")

        # 6) Execute
        corrected = strip_code_fences(corrected)
        log("Step 6: Executing final Cypher query:")
        log(corrected)

        try:
            rows = enhanced_graph.query(corrected)
        except Exception as e:
            # This catches runtime errors (e.g. bad property access) and returns "no answer"
            return no_answer(f"Execution error: {e}")

        log(f"Step 6: Query returned {len(rows)} row(s)")
        log("=== text2cypher_pipeline: END ===")

        return corrected, rows

    except CypherSyntaxError as e:
        # Extra safety net, in case something slips through
        return no_answer(f"Uncaught CypherSyntaxError: {e}")
    except Exception as e:
        # Very generic fallback, if something unexpected happens in the pipeline
        return no_answer(f"Unexpected error: {e}")
