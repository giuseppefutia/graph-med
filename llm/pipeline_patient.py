import ast
from datetime import datetime, date
from datetime import date as DateType
from typing import Any, Dict, Optional, List

from langchain_neo4j import Neo4jGraph

from util.config_loader import load_neo4j_config

_neo4j = load_neo4j_config()
enhanced_graph = Neo4jGraph(
    url=_neo4j["url"],
    username=_neo4j["username"],
    password=_neo4j["password"],
    database=_neo4j["database"],
    enhanced_schema=True,
)


def _parse_python_list_string(value: Optional[str]):
    """Parse stringified Python lists like "['G57.1', ...]" or "[{...}, ...]" safely."""
    if not value or not isinstance(value, str):
        return None
    try:
        return ast.literal_eval(value)
    except Exception:
        return None


def _parse_encounter_start(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        # Handles 'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM:SS'
        return datetime.fromisoformat(value).date()
    except Exception:
        try:
            # Fallback: first 10 chars as 'YYYY-MM-DD'
            return datetime.strptime(value[:10], "%Y-%m-%d").date()
        except Exception:
            return None


def get_patient_views(
    patient_id: str,
    encounter_date: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch virtualized 'patient' resources via APOC DV and return a list of
    normalized views.

    encounter_date semantics:
      - None       -> return ALL encounters (no date-based filtering).
      - "latest"   -> return ONLY the most recent encounter by Encounter.period.start.
      - "YYYY-MM-DD" -> return ONLY encounters whose Encounter.period.start
                        matches that calendar day.
    """
    cypher = """
    CALL apoc.dv.query('encounter', {patientId: $patient_id}) YIELD node AS v
    RETURN v
    """
    rows = enhanced_graph.query(cypher, {"patient_id": patient_id})
    if not rows:
        return []

    # Interpret encounter_date
    selection_mode: str = "all"  # "all" | "date" | "latest"
    requested_date: Optional[DateType] = None

    if encounter_date:
        enc_str = encounter_date.strip().lower()
        if enc_str == "latest":
            selection_mode = "latest"
        else:
            try:
                requested_date = DateType.fromisoformat(encounter_date.strip())
                selection_mode = "date"
            except ValueError:
                # If parsing fails, treat as "all" (safe fallback)
                selection_mode = "all"
                requested_date = None

    views: List[Dict[str, Any]] = []

    # Helper to build a normalized view from a virtual node
    def _build_view(
        props: Dict[str, Any],
        identity: Any,
        labels: List[str],
        element_id: Optional[str],
        encounter_start_str: Optional[str],
        encounter_start_date: Optional[DateType],
    ) -> Dict[str, Any]:
        icd_codes = _parse_python_list_string(props.get("ICD10_Codes"))
        ner_entities = _parse_python_list_string(props.get("NER_Entities"))
        ned_entities = _parse_python_list_string(props.get("NED_Entities"))

        return {
            "patient_id": props.get("PatientId") or props.get("patientId"),
            "identity": identity,
            "labels": labels,
            "elementId": element_id,

            # High-level clinical fields
            "condition": props.get("Condition"),
            "chief_complaint": props.get("ChiefComplaint"),
            "course_trend": props.get("CourseTrend"),
            "comorbidities": props.get("Comorbidities") or "",
            "plan_followup": props.get("Plan/FollowUp"),
            "medication_statement": props.get("MedicationStatement"),
            "notes": props.get("Notes"),

            # Encounter info
            "encounter": {
                "id": props.get("EncounterID"),
                "period_start": encounter_start_str,
                "reason_code": props.get("Encounter.reasonCode"),
                "class": props.get("Encounter.class"),
                "discharge_disposition": props.get(
                    "Encounter.hospitalization.dischargeDisposition"
                ),
                "diagnosis_rank": props.get("Encounter.diagnosis.rank"),
            },

            # Narrative + observations
            "narrative": props.get("Narrative"),
            "observation_vitals": props.get("Observation[vitals]"),
            "observation_text": props.get("Observation[key]"),
            "diagnostic_report": props.get("DiagnosticReport"),
            "procedure": props.get("Procedure"),

            # NLP/ICD-related
            "icd10_codes": icd_codes,
            "ner_entities": ner_entities,
            "ned_entities": ned_entities,

            # Traceability
            "filters": {
                "selection_mode": selection_mode,
                "encounter_date_requested": encounter_date,
                "encounter_start_date_parsed": (
                    encounter_start_date.isoformat()
                    if encounter_start_date
                    else None
                ),
            },
        }

    # --- Mode: "latest" -> choose the single latest encounter ---
    if selection_mode == "latest":
        chosen = None
        chosen_parsed_date: Optional[DateType] = None
        chosen_meta = None  # (props, identity, labels, element_id, start_str)

        for row in rows:
            v = row.get("v")

            if isinstance(v, dict) and "properties" in v:
                props = v.get("properties", {})
                identity = v.get("identity")
                labels = v.get("labels", [])
                element_id = v.get("elementId") or v.get("element_id")
            else:
                props = v if isinstance(v, dict) else {}
                identity = None
                labels = []
                element_id = None

            encounter_start_str = props.get("Encounter.period.start")
            encounter_start_date = _parse_encounter_start(encounter_start_str)

            if chosen is None:
                chosen = row
                chosen_parsed_date = encounter_start_date
                chosen_meta = (props, identity, labels, element_id, encounter_start_str)
                continue

            if encounter_start_date and (
                chosen_parsed_date is None or encounter_start_date > chosen_parsed_date
            ):
                chosen = row
                chosen_parsed_date = encounter_start_date
                chosen_meta = (props, identity, labels, element_id, encounter_start_str)

        if chosen_meta is None:
            return []

        props, identity, labels, element_id, encounter_start_str = chosen_meta
        encounter_start_date = chosen_parsed_date

        views.append(
            _build_view(
                props, identity, labels, element_id,
                encounter_start_str, encounter_start_date
            )
        )
        return views

    # --- Mode: "date" -> keep only encounters that match requested_date ---
    for row in rows:
        v = row.get("v")

        if isinstance(v, dict) and "properties" in v:
            props = v.get("properties", {})
            identity = v.get("identity")
            labels = v.get("labels", [])
            element_id = v.get("elementId") or v.get("element_id")
        else:
            props = v if isinstance(v, dict) else {}
            identity = None
            labels = []
            element_id = None

        encounter_start_str = props.get("Encounter.period.start")
        encounter_start_date = _parse_encounter_start(encounter_start_str)

        if selection_mode == "date":
            # Strict: require parsed date and exact match
            if not (encounter_start_date and encounter_start_date == requested_date):
                continue

        # selection_mode == "all" just falls through and includes everything
        views.append(
            _build_view(
                props, identity, labels, element_id,
                encounter_start_str, encounter_start_date
            )
        )

    return views
