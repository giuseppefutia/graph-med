from __future__ import annotations

import asyncio
import json
import logging
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple

from langchain_openai import ChatOpenAI

from util.config_loader import load_config_api
from llm.tool import build_ontology_mapper_tool


logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message="Pydantic serializer warnings", category=UserWarning)


def ontology_mapper_factory(base_importer_cls, backend: str, config_path: str = "config.ini"):
    """
    Factory returning a concrete OntologyMapper bound to `base_importer_cls` and `backend`.
    """

    async def _abatch_llm(tool, payloads: List[Dict[str, Any]], max_concurrency: int = 8):
        return await tool.abatch(payloads, config={"max_concurrency": max_concurrency})

    @dataclass
    class Candidate:
        id: str
        label: str
        score: float

    @dataclass
    class MappingDecision:
        best_id: Optional[str]
        best_label: Optional[str]
        confidence: float
        rationale: str
        support: Dict[str, Any]

    class OntologyMapper(base_importer_cls):
        """ICD → HPO mapping: select candidates, build context, disambiguate, write edges."""

        # Uses the stored embedding_label on each IcdDisease node instead
        # of calling the embedding API — no external API required.
        CYPHER_QUERY_TOPK_STORED_BATCH = """
        UNWIND $items AS row
        CALL (row) {
          MATCH (src:IcdDisease {id: row.id})
          WHERE src.embedding_label IS NOT NULL
          CALL db.index.vector.queryNodes($index, $k, src.embedding_label) YIELD node, score
          RETURN collect({id: node.id, label: node.label, score: score}) AS topk
        }
        RETURN row.key AS key, topk[0..$k] AS topk
        """

        CYPHER_SOURCE_CONTEXT = """
        MATCH (d:IcdDisease {id: $id})
        OPTIONAL MATCH (g:IcdGroup)-[:GROUP_HAS_DISEASE]->(d)
        OPTIONAL MATCH (c:IcdChapter)-[:CHAPTER_HAS_DISEASE]->(d)
        WITH d, collect(DISTINCT g.groupName) AS gnames, collect(DISTINCT c.chapterName) AS cnames
        RETURN {
          id: d.id,
          name: d.label,
          parentName: d.parentLabel,
          group:   { groupName: head(gnames) },
          chapter: { chapterName: head(cnames) }
        } AS context
        """

        CYPHER_SOURCE_CONTEXT_BATCH = """
        UNWIND $ids AS id
        MATCH (d:IcdDisease {id: id})
        OPTIONAL MATCH (g:IcdGroup)-[:GROUP_HAS_DISEASE]->(d)
        OPTIONAL MATCH (c:IcdChapter)-[:CHAPTER_HAS_DISEASE]->(d)
        WITH d, collect(DISTINCT g.groupName) AS gnames, collect(DISTINCT c.chapterName) AS cnames
        RETURN d.id AS id, {
          id: d.id,
          name: d.label,
          parentName: d.parentLabel,
          group:   { groupName: head(gnames) },
          chapter: { chapterName: head(cnames) }
        } AS context
        """

        CYPHER_CANDIDATE_CONTEXT = """
        MATCH (p:HpoPhenotype {id: $id})
        RETURN {
          id: p.id,
          label: p.label,
          exactSynonym: p.hasExactSynonym,
          description: p.comment
        } AS context
        """

        CYPHER_CANDIDATE_CONTEXT_BATCH = """
        UNWIND $ids AS id
        MATCH (p:HpoPhenotype {id: id})
        RETURN p.id AS id, {
          id: p.id,
          label: p.label,
          exactSynonym: p.hasExactSynonym,
          description: p.comment
        } AS context
        """

        def __init__(self):
            super().__init__()
            self.backend = backend

            # Tunables / defaults
            self.batch_size: int = 32
            self.k: int = 5
            self.llm_threshold: float = 0.7
            self.write_relationships: bool = True
            self.mapper_task: str = "icd_to_hpo_phenotype"

            # Labels/rel config
            self.source_label: str = "IcdDisease"
            self.target_label: str = "HpoPhenotype"
            self.target_index_name: str = "hpo_phenotype_embedding"
            self.relationship_type: str = "ICD_MAPS_TO_HPO_BY_EMBEDDING"
            self.relationship_conf_prop: str = "confidence"

            # LLM + tool
            url_llm = load_config_api("llm", path=config_path)
            self.llm = ChatOpenAI(
                api_key="EMPTY",
                base_url=url_llm,
                model_name="google/medgemma-4b-it",
                temperature=0,
                max_tokens=8192,
                top_p=0.9,
                stop=["<end_of_turn>", "</s>", "\nUser:", "\n\nUser:"],
                frequency_penalty=0.2,
                presence_penalty=0.0,
            )
            self.ontology_mapper_tool = build_ontology_mapper_tool(self.llm)

        @staticmethod
        def _md_to_params(md: MappingDecision) -> Dict[str, Any]:
            """Neo4j param mapping for MappingDecision (JSON-encode support)."""
            data = asdict(md)
            data["support"] = json.dumps(data.get("support", {}), ensure_ascii=False)
            return {k: v for k, v in data.items() if v is not None}

        # ──────────────────────────────────────────────────────────────
        # Candidate Selection (single & batch)
        # ──────────────────────────────────────────────────────────────
        def select_candidates_stored_in_batch(
            self, sources: List[Tuple[str, str]]  # [(source_id, source_label)]
        ) -> Dict[str, List[Candidate]]:
            """
            Path B variant of select_candidates_in_batch.

            Uses the stored embedding_label on each IcdDisease node as the query
            vector — no embedding API call required.
            Returns the same {label -> [Candidate]} structure for compatibility.
            """
            if not sources:
                return {}
            items = [{"id": sid, "key": lbl} for sid, lbl in sources]
            out: Dict[str, List[Candidate]] = {}
            with self._driver.session(database=self._database) as session:
                result = session.run(
                    self.CYPHER_QUERY_TOPK_STORED_BATCH,
                    items=items,
                    index=self.target_index_name,
                    k=self.k,
                )
                for record in result:
                    out[record["key"]] = [Candidate(**c) for c in record["topk"]]
            return out

        # ──────────────────────────────────────────────────────────────
        # Context Building (single & batch)
        # ──────────────────────────────────────────────────────────────
        def build_context(self, source_id: str, candidates: List[Candidate]) -> Dict[str, Any]:
            """Fetch source + candidate contexts for a single source/candidate set."""
            if self.mapper_task != "icd_to_hpo_phenotype":
                return {"source": {}, "candidates": []}

            with self._driver.session(database=self._database) as session:
                rec = session.run(self.CYPHER_SOURCE_CONTEXT, id=source_id).single()
                source_ctx = rec["context"] if rec else {}

            cand_ctxs: List[Dict[str, Any]] = []
            with self._driver.session(database=self._database) as session:
                for cand in candidates:
                    rec = session.run(self.CYPHER_CANDIDATE_CONTEXT, id=cand.id).single()
                    cand_ctxs.append(rec["context"] if rec else {})

            return {"source": source_ctx, "candidates": cand_ctxs}

        def build_context_in_batch(
            self,
            sources: List[Tuple[str, str]],        # [(source_id, source_label)]
            cand_map: Dict[str, List[Candidate]],  # label -> [Candidate, ...]
        ) -> Dict[str, Dict[str, Any]]:
            """
            Return:
              {
                source_id: {
                  "source": {...},
                  "candidates": [ {...}, ... ]
                },
                ...
              }
            """
            if not sources:
                return {}

            source_ids = [sid for sid, _ in sources]

            # 1) source contexts
            src_ctx: Dict[str, Dict[str, Any]] = {}
            with self._driver.session(database=self._database) as session:
                for r in session.run(self.CYPHER_SOURCE_CONTEXT_BATCH, ids=source_ids):
                    src_ctx[r["id"]] = r["context"]

            # 2) candidate contexts (unique ids across batch)
            all_cand_ids = [c.id for _, lbl in sources for c in cand_map.get(lbl, [])]
            uniq_cand_ids = list(dict.fromkeys(all_cand_ids))

            cand_ctx_map: Dict[str, Dict[str, Any]] = {}
            if uniq_cand_ids:
                with self._driver.session(database=self._database) as session:
                    for r in session.run(self.CYPHER_CANDIDATE_CONTEXT_BATCH, ids=uniq_cand_ids):
                        cand_ctx_map[r["id"]] = r["context"]

            # 3) stitch per-source
            out: Dict[str, Dict[str, Any]] = {}
            for sid, label in sources:
                cand_list = cand_map.get(label, [])
                out[sid] = {
                    "source": src_ctx.get(sid, {}),
                    "candidates": [cand_ctx_map.get(c.id, {}) for c in cand_list],
                }
            return out

        # ──────────────────────────────────────────────────────────────
        # LLM Disambiguation (single & batch)
        # ──────────────────────────────────────────────────────────────
        def _to_llm_payload(
            self,
            source_concept: str,
            source_context: Tuple[str, Any],
            candidate_contexts: List[Tuple[str, Any]],
        ) -> Dict[str, Any]:
            """LLM tool expects JSON strings for structured fields."""
            return {
                "source_concept": source_concept,
                "source_context": json.dumps(source_context, ensure_ascii=False),
                "candidate_list": json.dumps(candidate_contexts, ensure_ascii=False),
            }

        def disambiguate_candidates(
            self,
            source_concept: str,
            source_context: Tuple[str, Any],
            candidates: List[Tuple[str, Any]],
        ) -> MappingDecision:
            """Call the LLM tool once and parse into MappingDecision."""
            payload = self._to_llm_payload(source_concept, source_context, candidates)
            response = self.ontology_mapper_tool.invoke(payload)
            try:
                return MappingDecision(
                    best_id=response.get("best_id"),
                    best_label=response.get("best_label"),
                    confidence=float(response.get("confidence", 0.0)),
                    rationale=response.get("rationale", "") or "",
                    support=response.get("support") or {},
                )
            except Exception as exc:
                logging.error("Failed to parse LLM response: %s", exc)
                return MappingDecision(None, None, 0.0, "Failed to parse LLM response.", {})

        def disambiguate_candidates_batch_sync(
            self,
            items: Iterable[Tuple[str, Tuple[str, Any], List[Tuple[str, Any]]]],  # (source_concept, source_context, candidate_contexts)
            max_concurrency: int = 8,
        ) -> List[MappingDecision]:
            """Batch LLM disambiguation. Returns one MappingDecision per input (same order)."""
            payloads = [
                self._to_llm_payload(source_concept, source_context, candidates)
                for source_concept, source_context, candidates in items
            ]

            raw_responses = asyncio.run(_abatch_llm(self.ontology_mapper_tool, payloads, max_concurrency))

            decisions: List[MappingDecision] = []
            for resp in raw_responses:
                try:
                    decisions.append(
                        MappingDecision(
                            best_id=resp.get("best_id"),
                            best_label=resp.get("best_label"),
                            confidence=float(resp.get("confidence", 0.0)),
                            rationale=resp.get("rationale", "") or "",
                            support=resp.get("support") or {},
                        )
                    )
                except Exception as exc:
                    logging.error("Failed to parse LLM response: %s", exc)
                    decisions.append(MappingDecision(None, None, 0.0, "Failed to parse LLM response.", {}))
            return decisions

        # ──────────────────────────────────────────────────────────────
        # End-to-end runners
        # ──────────────────────────────────────────────────────────────
        def run_disambiguation(self, source_id: str, source_label: str) -> MappingDecision:
            """Single source disambiguation: select → context → LLM."""
            candidates = self.select_candidates_stored_in_batch(
                [(source_id, source_label)]
            ).get(source_label, [])
            ctx = self.build_context(source_id=source_id, candidates=candidates)
            result = self.disambiguate_candidates(
                source_concept=source_label,
                source_context=ctx.get("source", {}),
                candidates=ctx.get("candidates", []),
            )
            logging.debug("Disambiguation result for '%s': %s", source_label, result)
            return result

        def run_disambiguation_in_batch(
            self,
            sources: List[Tuple[str, str]],  # [(source_id, source_label)]
            max_concurrency: int = 8,
        ) -> List[Tuple[str, str, MappingDecision]]:
            """Batch disambiguation. Keeps input order in output triplets."""
            if not sources:
                return []

            # 1) vector candidates (stored embeddings — no API call)
            cand_map = self.select_candidates_stored_in_batch(sources)

            # 2) graph contexts
            ctx_map = self.build_context_in_batch(sources, cand_map)

            # 3) LLM inputs (preserve order)
            items = []
            for sid, lbl in sources:
                sc = ctx_map.get(sid, {}).get("source", {})
                cc = ctx_map.get(sid, {}).get("candidates", [])
                items.append((lbl, sc, cc))

            # 4) LLM batch
            decisions = self.disambiguate_candidates_batch_sync(items, max_concurrency=max_concurrency)

            # 5) align back
            return [(sid, lbl, md) for (sid, lbl), md in zip(sources, decisions)]

        # ──────────────────────────────────────────────────────────────
        # Orchestration: read, filter, and write
        # ──────────────────────────────────────────────────────────────
        def test(self) -> None:
            """Quick smoke test for single + batch runs (does not write to DB)."""
            source = ("R55", "Syncope and collapse")
            res = self.run_disambiguation(source_id=source[0], source_label=source[1])
            print(f"### Single: '{source[1]}' → {res.best_id} ({res.best_label}), conf={res.confidence}")

            sources = [("R55", "Syncope and collapse"), ("G44.1", "Vascular headache")]
            for sid, lbl, md in self.run_disambiguation_in_batch(sources):
                print(f"### Batch:  '{lbl}' → {md.best_id} ({md.best_label}), conf={md.confidence}")

        def count_missing_source_nodes(self) -> int:
            """Count source nodes not yet processed by this mapper."""
            query = f"""
            MATCH (n:{self.source_label})
            WHERE NOT "ProcessedWithOntologyMapper" IN labels(n)
            RETURN count(n) AS cnt
            """
            with self._driver.session(database=self._database) as session:
                rec = session.run(query).single()
                return rec["cnt"] if rec else 0

        def get_source_nodes(self) -> Generator[Dict[str, Any], None, None]:
            """
            Stream source nodes (single-disambiguation path).
            Prefer `get_source_nodes_in_batch` for efficiency.
            """
            query = f"""
            MATCH (n:{self.source_label})
            WHERE NOT n:ProcessedWithOntologyMapper
            RETURN n.id AS id, coalesce(n.label, n.name) AS label
            """
            with self._driver.session(database=self._database) as session:
                for record in session.run(query):
                    md = self.run_disambiguation(source_id=record["id"], source_label=record["label"])
                    if md.confidence >= self.llm_threshold:
                        yield {
                            "id": record["id"],
                            "label": record["label"],
                            "disambiguation_result": self._md_to_params(md),
                        }

        def get_source_nodes_in_batch(self, batch_size: int = 8) -> Generator[Dict[str, Any], None, None]:
            """Stream nodes but perform selection/context/LLM steps in batches."""
            query = f"""
            MATCH (n:{self.source_label})
            WHERE NOT "ProcessedWithOntologyMapper" IN labels(n)
            RETURN n.id AS id, n.label AS label
            """

            with self._driver.session(database=self._database) as session:
                rows: List[Tuple[str, str]] = [(rec["id"], rec["label"]) for rec in session.run(query)]

            for i in range(0, len(rows), batch_size):
                chunk = rows[i : i + batch_size]
                for sid, lbl, md in self.run_disambiguation_in_batch(chunk, max_concurrency=8):
                    if md.confidence >= self.llm_threshold:
                        yield {
                            "id": sid,
                            "label": lbl,
                            "disambiguation_result": self._md_to_params(md),
                        }

        def merge_mapping_relationship(self) -> None:
            """Write mapping edges and mark sources as processed."""
            logging.info(
                "Merging mapping relationship: %s -> %s [%s]",
                self.source_label,
                self.target_label,
                self.relationship_type,
            )
            query = f"""
            UNWIND $batch AS item
            MATCH (source:{self.source_label} {{id: item.id}})
            MATCH (target:{self.target_label} {{id: item.disambiguation_result.best_id}})
            MERGE (source)-[r:{self.relationship_type}]->(target)
            SET r.{self.relationship_conf_prop} = item.disambiguation_result.confidence,
                r.rationale = item.disambiguation_result.rationale,
                r.support = item.disambiguation_result.support
            SET source:ProcessedWithOntologyMapper
            """

            total = self.count_missing_source_nodes()
            # get_source_nodes_in_batch streams items lazily, already batched internally.
            self.batch_store(query, self.get_source_nodes_in_batch(), size=total)

        def apply_updates(self) -> None:
            """Entrypoint used by the CLI importer."""
            logging.info("Testing Ontology Mapper...")
            self.test()
            print("\n\n\n")
            logging.info("Starting Ontology Mapping import...")
            self.merge_mapping_relationship()

    return OntologyMapper


if __name__ == "__main__":
    from util.cli_entry import run_backend_importer

    run_backend_importer(
        ontology_mapper_factory,
        description="Run Ontology Mapper.",
        file_help="No file needed for ontology mapping.",
        default_base_path="./data/",
        require_file=False,
    )
