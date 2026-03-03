import logging
import time
from typing import Sequence, List, Dict

from neo4j.exceptions import ClientError as Neo4jClientError

from util.config_loader import load_config_api
from util.api_client import ApiClient


def hpo_embedding_importer_factory(base_importer_cls, backend: str, config_path: str = "config.ini"):

    class HpoEmbeddingImporter(base_importer_cls):

        def __init__(self):
            super().__init__()
            self.backend = backend
            self.cfg = load_config_api("embedding", path=config_path)
            self.api = ApiClient(self.cfg)

            self.node_specs: List[Dict] = [
                {
                    "label": "HpoPhenotype",
                    "id_prop": "id",
                    "text_prop": "label",
                    "embed_prop": "embedding_label",
                    "index_name": "hpo_phenotype_embedding",
                    "dim": 3584,
                    "similarity": "cosine",
                    "log_tag": "HPO-Phen"
                },
                {
                    "label": "HpoDisease",
                    "id_prop": "id",
                    "text_prop": "label",
                    "embed_prop": "embedding_label",
                    "index_name": "hpo_disease_embedding",
                    "dim": 3584,
                    "similarity": "cosine",
                    "log_tag": "HPO-Disease"
                },
            ]

        # ──────────────────────────────────────────────────────────────
        # Helpers: vector index & embedding API
        # ──────────────────────────────────────────────────────────────
        def _ensure_vector_index(
            self, label: str, embed_prop: str, index_name: str, dim: int = 3584, similarity: str = "cosine"
        ):
            query = f"""
            CREATE VECTOR INDEX {index_name} IF NOT EXISTS
            FOR (n:{label})
            ON (n.{embed_prop})
            OPTIONS {{
                IndexConfig: {{
                    `vector.dimensions`: {dim},
                    `vector.similarity_function`: '{similarity}'
                }}
            }};
            """
            with self._driver.session(database=self._database) as session:
                try:
                    session.run(query)
                except Neo4jClientError as e:
                    if e.code != "Neo.ClientError.Schema.EquivalentSchemaRuleAlreadyExists":
                        raise

        def _embed_label(self, text: str) -> Sequence[float]:
            """Kept for compatibility."""
            resp = self.api.post('/v1/embeddings', {'input': text})
            return resp[0]["data"][0]["embedding"]

        def _embed_labels(self, texts: List[str]) -> List[Sequence[float]]:
            if not texts:
                return []
            resp = self.api.post('/v1/embeddings', {'input': texts})
            vectors = [item["embedding"] for item in resp[0]["data"]]
            if len(vectors) != len(texts):
                raise RuntimeError(f"Embedding count mismatch ({len(vectors)} != {len(texts)})")
            return vectors

        def _count_missing(self, label: str, embed_prop: str) -> int:
            query = f"""
            MATCH (n:{label})
            WHERE n.{embed_prop} IS NULL
            RETURN count(n) AS cnt
            """
            with self._driver.session(database=self._database) as session:
                rec = session.run(query).single()
                return rec["cnt"] if rec else 0

        # ──────────────────────────────────────────────────────────────
        # Batch embedding pipeline
        # ──────────────────────────────────────────────────────────────
        def _add_embeddings_for(
            self,
            *,
            label: str,
            id_prop: str = "id",
            text_prop: str = "label",
            embed_prop: str = "embedding_label",
            batch_size: int = 128,
            log_tag: str = "Embed"
        ):
            fetch_query = f"""
            MATCH (n:{label})
            WHERE n.{embed_prop} IS NULL AND n.{text_prop} IS NOT NULL
            RETURN n.{id_prop} AS id, n.{text_prop} AS text
            """

            write_query = f"""
            UNWIND $rows AS row
            MATCH (n:{label} {{{id_prop}: row.id}})
            SET n.{embed_prop} = row.embedding
            """

            total = self._count_missing(label, embed_prop)
            if total == 0:
                logging.info(f"[{log_tag}] No missing embeddings for :{label}.")
                return

            processed = 0
            start_ts = time.time()
            batch_idx = 0

            def log_progress():
                elapsed = max(time.time() - start_ts, 1e-6)
                rate = processed / elapsed
                pct = (processed / total) * 100 if total else 100.0
                remaining = max(total - processed, 0)
                eta = remaining / rate if rate > 0 else float('inf')
                logging.info(
                    f"[{log_tag}] Batch {batch_idx} | "
                    f"{processed}/{total} ({pct:.1f}%) | "
                    f"{rate:.1f} nodes/s | ETA ~ {int(eta)}s"
                )

            with self._driver.session(database=self._database) as session:
                result = session.run(fetch_query)

                buffer_ids: List[str] = []
                buffer_texts: List[str] = []

                def flush():
                    nonlocal processed, batch_idx
                    if not buffer_texts:
                        return
                    embeddings = self._embed_labels(buffer_texts)
                    rows = [{"id": i, "embedding": e} for i, e in zip(buffer_ids, embeddings)]
                    session.run(write_query, rows=rows)
                    processed += len(rows)
                    batch_idx += 1
                    buffer_ids.clear()
                    buffer_texts.clear()
                    log_progress()

                for rec in result:
                    node_id = rec["id"]
                    text = rec["text"]
                    if not text:
                        continue
                    buffer_ids.append(node_id)
                    buffer_texts.append(text)
                    if len(buffer_texts) >= batch_size:
                        flush()

                flush()  # remainder

            elapsed = max(time.time() - start_ts, 1e-6)
            logging.info(f"[{log_tag}] Completed: {processed}/{total} in {int(elapsed)}s "
                         f"({processed/elapsed:.1f} nodes/s)")

        # ──────────────────────────────────────────────────────────────
        # Orchestration
        # ──────────────────────────────────────────────────────────────
        def apply_updates(self, batch_size: int = 128):
            logging.info("Ensuring vector indexes...")
            for spec in self.node_specs:
                self._ensure_vector_index(
                    label=spec["label"],
                    embed_prop=spec["embed_prop"],
                    index_name=spec["index_name"],
                    dim=spec.get("dim", 3584),
                    similarity=spec.get("similarity", "cosine"),
                )

            logging.info("Embedding missing labels...")
            for spec in self.node_specs:
                self._add_embeddings_for(
                    label=spec["label"],
                    id_prop=spec["id_prop"],
                    text_prop=spec["text_prop"],
                    embed_prop=spec["embed_prop"],
                    batch_size=batch_size,
                    log_tag=spec.get("log_tag", spec["label"]),
                )

    return HpoEmbeddingImporter


if __name__ == '__main__':
    from util.cli_entry import run_backend_importer

    run_backend_importer(
        hpo_embedding_importer_factory,
        description="Run HPO Embedding.",
        file_help="No file needed for HPO embedding.",
        default_base_path="./data/",
        require_file=False
    )