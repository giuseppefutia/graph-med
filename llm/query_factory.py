import time
from typing import List, Dict, Any, Optional, Set

from langchain_neo4j import Neo4jGraph
from neo4j.exceptions import ServiceUnavailable, SessionExpired

from util.config_loader import load_neo4j_config

_neo4j = load_neo4j_config()
enhanced_graph = Neo4jGraph(
    url=_neo4j["url"],
    username=_neo4j["username"],
    password=_neo4j["password"],
    database=_neo4j["database"],
    enhanced_schema=True,
)


def _run_query(
    cypher: str, params: Optional[Dict[str, Any]] = None, retries: int = 2
):
    """Run a Cypher query with minimal retry on transient errors."""
    params = params or {}
    attempt = 0
    while True:
        try:
            return enhanced_graph.query(cypher, params)
        except (ServiceUnavailable, SessionExpired):
            if attempt >= retries:
                raise
            attempt += 1
            time.sleep(0.5 * attempt)
        except Exception:
            raise


def get_patient_icd_codes(patient_id: str) -> List[str]:
    """Fetch *all* ICD-10 codes for a patient across all rows.

    Handles BOM-prefixed property names and returns a sorted list of
    unique, uppercase, non-empty ICD-10 codes.
    """
    cypher = """
    CALL apoc.dv.query('encounter', {patientId: $pid}) YIELD node AS v
    WITH apoc.convert.fromJsonList(
           coalesce(
               apoc.any.property(v, 'ICD10_Codes'),
               apoc.any.property(v, '\uFEFFICD10_Codes')
           )
         ) AS codes
    RETURN codes
    """

    rows = _run_query(cypher, {"pid": patient_id})
    if not rows:
        return []

    all_codes: Set[str] = set()
    for row in rows:
        row_codes = row.get("codes") or []
        for c in row_codes:
            cleaned = (c or "").strip().upper()
            if cleaned:
                all_codes.add(cleaned)

    return sorted(all_codes)


def map_icd_to_hpo(icd_codes: List[str]) -> List[str]:
    """Map ICD codes to HPO phenotype IDs."""
    if not icd_codes:
        return []

    cypher = """
    CALL () {
        UNWIND $codes AS code
        MATCH (:IcdDisease {id: code})-[:ICD_MAPS_TO_HPO_BY_EMBEDDING]->(h:HpoPhenotype)
        RETURN DISTINCT h.id AS hpo_id

        UNION

        UNWIND $codes AS code
        MATCH (:IcdDisease {id: code})<-[:UMLS_TO_ICD]-(:Umls)-[:UMLS_TO_HPO_PHENOTYPE]->(h:HpoPhenotype)
        RETURN DISTINCT h.id AS hpo_id
    }
    RETURN collect(DISTINCT hpo_id) AS hpo_ids;
    """

    rows = _run_query(cypher, {"codes": icd_codes})
    return rows[0].get("hpo_ids", []) if rows else []


def rollup_hpo_to_ancestors(hpo_ids: List[str]) -> List[str]:
    """Roll up to ancestors (including self) over :SUBCLASSOF*0.. and return unique IDs."""
    if not hpo_ids:
        return []

    cypher = """
    UNWIND $ids AS hid
    MATCH (h:HpoPhenotype {id: hid})-[:subClassOf*0..]->(anc:HpoPhenotype)
    WITH collect(DISTINCT anc.id) AS target
    RETURN [id IN apoc.coll.toSet(target) WHERE id IS NOT NULL] AS target
    """

    rows = _run_query(cypher, {"ids": hpo_ids})
    return rows[0].get("target", []) if rows else []


def compute_coverage(target_ids: List[str], limit: int = 20) -> List[Dict[str, Any]]:
    """Given rolled-up HPO target IDs, compute coverage across diseases."""
    if not target_ids:
        return []

    cypher = """
    MATCH (d:HpoDisease)-[:HAS_PHENOTYPIC_FEATURE]->(dh:HpoPhenotype)
    WHERE dh.id IN $target
    WITH d, collect(DISTINCT dh.id) AS got

    MATCH (d)-[:HAS_PHENOTYPIC_FEATURE]->(all_dh:HpoPhenotype)
    WITH d,
        apoc.coll.toSet(collect(DISTINCT all_dh.id)) AS existing,
        apoc.coll.toSet(got)                         AS got

    WITH d, existing,
        apoc.coll.intersection(existing, got) AS overlap,
        apoc.coll.subtract(existing, got)     AS missing,
        got

    WITH d, size(overlap) AS covered,
        size(existing)   AS total,
        missing

    RETURN d.id   AS diseaseId,
        d.label AS diseaseName,
        covered,
        total,
        round(100.0 * covered / total, 1) AS coveragePct,
        missing AS missingHpoIds
    ORDER BY covered DESC
    LIMIT $limit;
    """

    return _run_query(cypher, {"target": target_ids, "limit": int(limit)})


def rank_diseases_for_patient(
    patient_id: str,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """End-to-end pipeline using the canonical helper functions:
    patient -> ICD codes -> HPO phenotypes -> ancestors -> coverage.
    """
    # 1) ICD codes from DV
    icd_codes = get_patient_icd_codes(patient_id)
    if not icd_codes:
        return []

    # 2) ICD -> HPO
    hpo_ids = map_icd_to_hpo(icd_codes)
    if not hpo_ids:
        return []

    # 3) Roll up to ancestors
    target_ids = rollup_hpo_to_ancestors(hpo_ids)
    if not target_ids:
        return []

    # 4) Compute coverage
    return compute_coverage(target_ids, limit=limit)
