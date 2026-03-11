import re
from typing import Any, Dict, List, Optional, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from llm.chain import (
    get_guardrails_chain,
    get_final_answer_chain,
)

from llm.tool import build_patient_info_tool

from llm.query_factory import (
    get_patient_icd_codes,
    map_icd_to_hpo,
    rollup_hpo_to_ancestors,
    compute_coverage,
)

from llm.pipeline import text2cypher_pipeline


class AgentState(TypedDict, total=False):
    question: str
    patient_id: Optional[str]
    icd_codes: List[str]
    hpo_ids: List[str]
    target_ids: List[str]
    results: Any
    steps: List[str]
    mode: str  # "stepwise", "text2cypher", or "patient_info"
    error: Optional[str]
    final_answer: str


############################
### Node implementations ###
############################

llm = ChatOpenAI(
    api_key="EMPTY",
    base_url="https://chatcompletion-uncognizable-nilda.ngrok-free.dev/v1",
    model_name="google/medgemma-4b-it",
    temperature=0,
    max_tokens=24000,
    top_p=0.9,
    stop=["<end_of_turn>", "</s>", "\nUser:", "\n\nUser:"],
    frequency_penalty=0.2,
    presence_penalty=0.0,
)

patient_info_tool = build_patient_info_tool(llm)


def node_guardrails(state: AgentState) -> AgentState:
    guard = get_guardrails_chain(llm).invoke({"question": state.get("question")})
    if getattr(guard, "decision", None) == "end":
        return {
            **state,
            "steps": [*(state.get("steps") or []), "guardrails"],
            "results": "This question is not related to the application domain.",
            "final_answer": "This question is not related to the application domain.",
        }
    return {**state, "steps": [*(state.get("steps") or []), "guardrails"]}


def node_extract_inputs(state: AgentState) -> AgentState:
    q = state.get("question", "")
    pid = state.get("patient_id")
    if not pid:
        # try simple patterns e.g., patientId:'P001', patientId="P123", patient P001
        m = re.search(r"patientId\s*[:=]\s*['\"]([\w-]+)['\"]", q, re.IGNORECASE)
        if not m:
            m = re.search(r"\bpatient\s+([\w-]+)\b", q, re.IGNORECASE)
        pid = m.group(1) if m else None
    return {
        **state,
        "patient_id": pid,
        "steps": [*(state.get("steps") or []), "extract"],
    }


def node_get_icd(state: AgentState) -> AgentState:
    try:
        codes = (
            get_patient_icd_codes(state.get("patient_id") or "")
            if state.get("patient_id")
            else []
        )
        print(f"DEBUG: Retrieved ICD codes: {codes}")
        return {
            **state,
            "icd_codes": codes,
            "steps": [*(state.get("steps") or []), "get_icd"],
        }
    except Exception as e:
        return {
            **state,
            "error": f"ICD retrieval failed: {e}",
            "steps": [*(state.get("steps") or []), "get_icd"],
        }


def node_icd_to_hpo(state: AgentState) -> AgentState:
    try:
        hpos = map_icd_to_hpo(state.get("icd_codes") or [])
        return {
            **state,
            "hpo_ids": hpos,
            "steps": [*(state.get("steps") or []), "icd_to_hpo"],
        }
    except Exception as e:
        return {
            **state,
            "error": f"ICD→HPO mapping failed: {e}",
            "steps": [*(state.get("steps") or []), "icd_to_hpo"],
        }


def node_rollup(state: AgentState) -> AgentState:
    try:
        targets = rollup_hpo_to_ancestors(state.get("hpo_ids") or [])
        return {
            **state,
            "target_ids": targets,
            "steps": [*(state.get("steps") or []), "rollup"],
        }
    except Exception as e:
        return {
            **state,
            "error": f"Roll-up failed: {e}",
            "steps": [*(state.get("steps") or []), "rollup"],
        }


def node_patient_info(state: AgentState) -> AgentState:
    """Use the patient_info tool to explain the virtualized patient node."""
    pid = state.get("patient_id")
    if not pid:
        # Defensive fallback
        return {
            **state,
            "error": "patient_info node called without patient_id",
            "steps": [*(state.get("steps") or []), "patient_info"],
            "results": None,
            "final_answer": "No patient_id was found in the question, so I cannot retrieve patient data.",
            "mode": "patient_info",
        }

    tool_result = patient_info_tool.invoke({
        "patient_id": pid,
        "question": state.get("question", ""),
    })

    # tool_result["answer"] is already a clinician-ready "Answer: ..." block
    return {
        **state,
        "results": tool_result,
        "final_answer": tool_result.get("answer"),
        "mode": "patient_info",
        "steps": [*(state.get("steps") or []), "patient_info"],
    }


def node_coverage(state: AgentState) -> AgentState:
    try:
        results = compute_coverage(state.get("target_ids") or [], limit=20)
        return {
            **state,
            "results": results,
            "steps": [*(state.get("steps") or []), "coverage"],
        }
    except Exception as e:
        return {
            **state,
            "error": f"Coverage failed: {e}",
            "steps": [*(state.get("steps") or []), "coverage"],
        }


def node_fallback_text2cypher(state: AgentState) -> AgentState:
    """Run the monolithic path if stepwise path lacks required inputs."""
    cypher, recs = text2cypher_pipeline(llm, state.get("question", ""))
    return {
        **state,
        "mode": "text2cypher",
        "results": recs,
        "steps": [*(state.get("steps") or []), "text2cypher"],
        "_last_cypher": cypher,  # debug-only
    }


def node_finalize(state: AgentState) -> AgentState:
    # Let user's chain handle verbalization if available
    try:
        final = get_final_answer_chain(llm).invoke(
            {
                "question": state.get("question"),
                "results": state.get("results"),
            }
        )
        return {
            **state,
            "final_answer": final,
            "steps": [*(state.get("steps") or []), "finalize"],
        }
    except Exception:
        # Minimal default
        return {
            **state,
            "final_answer": str(state.get("results")),
            "steps": [*(state.get("steps") or []), "finalize"],
        }


#####################
### Graph wiring  ###
#####################

def build_graph() -> Any:
    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("guardrails", node_guardrails)
    graph.add_node("extract", node_extract_inputs)
    graph.add_node("get_icd", node_get_icd)
    graph.add_node("icd_to_hpo", node_icd_to_hpo)
    graph.add_node("rollup", node_rollup)
    graph.add_node("patient_info", node_patient_info)
    graph.add_node("coverage", node_coverage)
    graph.add_node("fallback", node_fallback_text2cypher)
    graph.add_node("finalize", node_finalize)

    # Entry
    graph.set_entry_point("guardrails")

    # Edges (stepwise path)
    graph.add_edge("guardrails", "extract")

    # NEW: richer router after extract
    def route_after_extract(state: AgentState) -> str:
        q = (state.get("question") or "").lower()
        pid = state.get("patient_id")

        if not pid:
            # No patient_id → we can’t do patient-centric or ICD-centric path, use fallback
            return "fallback"

        # Heuristic keywords that suggest “explain this patient” rather than “phenotype coverage”
        patient_info_keywords = [
            "summarize",
            "summary",
            "presentation",
            "findings",
            "what is documented",
            "documented",
            "treatment",
            "therapy",
            "follow-up",
            "follow up",
            "vitals",
            "narrative",
            "exam",
            "examination",
            "diagnosis",
            "icd",
            "clinical picture",
            "course",
            "evolution",
            "nlp",
            "ner entities",
            "ned entities",
        ]

        if any(kw in q for kw in patient_info_keywords):
            return "patient_info"

        # Otherwise, assume this is your original ICU/HPO coverage-style question
        return "get_icd"

    graph.add_conditional_edges(
        "extract",
        route_after_extract,
        {
            "patient_info": "patient_info",
            "get_icd": "get_icd",
            "fallback": "fallback",
        },
    )

    # If no ICD codes found, fallback
    def have_icd(state: AgentState) -> str:
        return "icd_to_hpo" if state.get("icd_codes") else "fallback"

    graph.add_conditional_edges(
        "get_icd",
        have_icd,
        {"icd_to_hpo": "icd_to_hpo", "fallback": "fallback"},
    )

    # If no HPO found, fallback
    def have_hpo(state: AgentState) -> str:
        return "rollup" if state.get("hpo_ids") else "fallback"

    graph.add_conditional_edges(
        "icd_to_hpo",
        have_hpo,
        {"rollup": "rollup", "fallback": "fallback"},
    )

    # If no targets after roll-up, fallback
    def have_targets(state: AgentState) -> str:
        return "coverage" if state.get("target_ids") else "fallback"

    graph.add_conditional_edges(
        "rollup",
        have_targets,
        {"coverage": "coverage", "fallback": "fallback"},
    )

    # After coverage, finalize
    graph.add_edge("coverage", "finalize")

    # Fallback then finalize
    graph.add_edge("fallback", "finalize")

    # NEW: patient_info already produces a final clinician answer → go straight to END
    graph.add_edge("patient_info", END)

    # No checkpointer: simple in-memory, one-shot graph
    compiled = graph.compile()
    return compiled


#####################
### Run the Agent ###
#####################

def run_agent(question: str, *, patient_id: Optional[str] = None) -> Dict[str, Any]:
    """Convenience for invoking the agent programmatically.

    Example:
        run_agent("Rank diseases by HPO coverage for patientId:'P001'")
    """
    compiled = build_graph()
    initial: AgentState = {
        "question": question,
        "patient_id": patient_id,
        "steps": [],
        "mode": "stepwise",
    }
    final_state = compiled.invoke(initial)
    
    return {
        "steps": final_state.get("steps"),
        "mode": final_state.get("mode", "stepwise"),
        "patient_id": final_state.get("patient_id"),
        "icd_codes": final_state.get("icd_codes"),
        "hpo_ids": final_state.get("hpo_ids"),
        "target_ids": final_state.get("target_ids"),
        "results": final_state.get("results"),
        "final_answer": final_state.get("final_answer"),
    }


############################
### CLI entry (optional) ###
############################

if __name__ == "__main__":
    # Minimal smoke test; relies on your DB contents.
    import json

    # Original phenotype/coverage query
    q = "Show possible diseases by HPO coverage for patientId:'P003'. Report the covered, total, and percentage coverage."

    # NEW: patient-info style query
    # q = "For patient P003, summarize the diagnosis, key findings, and treatment that are documented in the data."

    # q = "Provide me details on the follow up plan of patient P003 in the latest encounter."

    out = run_agent(q)
    print(json.dumps(out, indent=2, ensure_ascii=False))

    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
