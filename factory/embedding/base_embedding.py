import logging
import time
from queue import Queue
from threading import Thread
from typing import Sequence, List, Dict

from neo4j.exceptions import ClientError as Neo4jClientError

from util.config_loader import load_config_api
from util.api_client import ApiClient

_SENTINEL = object()


def embedding_factory(
    base_importer_cls,
    backend: str,
    node_specs: List[Dict],
    config_path: str = "config.ini",
):
    """
    Generic embedding importer factory shared by all ontology-specific importers.

    Pipelines embedding API calls and Neo4j writes: a background writer thread
    handles DB writes while the main thread immediately starts the next embed
    call, so GPU inference and DB I/O overlap.

    node_specs entries must have:
        label, id_prop, text_prop, embed_prop, index_name
    Optional per-entry:
        dim (default 3584), similarity (default "cosine"), log_tag
    """

    class EmbeddingImporter(base_importer_cls):

        def __init__(self):
            super().__init__()
            self.backend = backend
            self.cfg = load_config_api("embedding", path=config_path)
            self.api = ApiClient(self.cfg)
            self.node_specs: List[Dict] = node_specs

        # ── Vector index ───────────────────────────────────────────────────

        def _ensure_vector_index(
            self, label: str, embed_prop: str, index_name: str,
            dim: int = 3584, similarity: str = "cosine",
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

        # ── Embedding API ──────────────────────────────────────────────────

        def _embed_labels(self, texts: List[str]) -> List[Sequence[float]]:
            if not texts:
                return []
            resp = self.api.post('/v1/embeddings', {'input': texts})
            vectors = [item["embedding"] for item in resp[0]["data"]]
            if len(vectors) != len(texts):
                raise RuntimeError(
                    f"Embedding count mismatch ({len(vectors)} vs {len(texts)})"
                )
            return vectors

        # ── Helpers ────────────────────────────────────────────────────────

        def _count_missing(self, label: str, embed_prop: str) -> int:
            query = f"""
            MATCH (n:{label})
            WHERE n.{embed_prop} IS NULL
            RETURN count(n) AS cnt
            """
            with self._driver.session(database=self._database) as session:
                rec = session.run(query).single()
                return rec["cnt"] if rec else 0

        # ── Pipelined batch embedding ──────────────────────────────────────

        def _add_embeddings_for(
            self, *,
            label: str,
            id_prop: str = "id",
            text_prop: str = "label",
            embed_prop: str = "embedding_label",
            batch_size: int = 128,
            log_tag: str = "Embed",
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

            n_batches = (total + batch_size - 1) // batch_size
            logging.info(
                f"[{log_tag}] Starting :{label} — {total} nodes, "
                f"batch_size={batch_size}, ~{n_batches} batches"
            )

            processed = 0
            start_ts = time.time()
            batch_idx = 0

            def log_progress(embed_s: float):
                elapsed = max(time.time() - start_ts, 1e-6)
                rate = processed / elapsed
                pct = (processed / total) * 100 if total else 100.0
                eta = max(total - processed, 0) / rate if rate > 0 else float("inf")
                logging.info(
                    f"[{log_tag}] Batch {batch_idx}/{n_batches} | "
                    f"{processed}/{total} ({pct:.1f}%) | "
                    f"embed {embed_s:.1f}s | "
                    f"{rate:.1f} nodes/s | ETA ~{int(eta)}s"
                )

            # Bounded queue (maxsize=1): the writer thread picks up each batch
            # as soon as it is ready, so the main thread can start the next
            # embed call immediately after enqueueing — overlapping GPU
            # inference with Neo4j writes.
            queue: Queue = Queue(maxsize=1)
            errors: List[Exception] = []

            def _writer():
                with self._driver.session(database=self._database) as ws:
                    while True:
                        item = queue.get()
                        if item is _SENTINEL:
                            break
                        try:
                            ws.run(write_query, rows=item)
                        except Exception as exc:
                            logging.error(f"[{log_tag}] Write error: {exc}")
                            errors.append(exc)
                            break

            writer = Thread(target=_writer, name=f"emb-writer-{label}", daemon=True)
            writer.start()

            try:
                with self._driver.session(database=self._database) as rs:
                    result = rs.run(fetch_query)
                    buffer_ids: List[str] = []
                    buffer_texts: List[str] = []

                    def flush():
                        nonlocal processed, batch_idx
                        if not buffer_texts or errors:
                            return
                        t0 = time.time()
                        embeddings = self._embed_labels(buffer_texts)
                        embed_s = time.time() - t0
                        rows = [
                            {"id": i, "embedding": e}
                            for i, e in zip(buffer_ids, embeddings)
                        ]
                        queue.put(rows)  # blocks until writer is ready for next batch
                        processed += len(rows)
                        batch_idx += 1
                        buffer_ids.clear()
                        buffer_texts.clear()
                        log_progress(embed_s)

                    for rec in result:
                        if errors:
                            break
                        text = rec["text"]
                        if not text:
                            continue
                        buffer_ids.append(rec["id"])
                        buffer_texts.append(text)
                        if len(buffer_texts) >= batch_size:
                            flush()

                    if not errors:
                        flush()  # remainder
            finally:
                queue.put(_SENTINEL)
                writer.join()

            if errors:
                raise errors[0]

            elapsed = max(time.time() - start_ts, 1e-6)
            logging.info(
                f"[{log_tag}] Done: {processed}/{total} in {int(elapsed)}s "
                f"({processed / elapsed:.1f} nodes/s)"
            )

        # ── Orchestration ──────────────────────────────────────────────────

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

    return EmbeddingImporter
