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

    Uses a 3-stage pipeline so that Neo4j reading, GPU embedding, and Neo4j
    writing all overlap:

        [reader] ──▶ input_q ──▶ [embedder-0] ──▶ output_q ──▶ [writer]
                              ──▶ [embedder-1] ──▶
                              ──▶ [embedder-2] ──▶

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

        def _embed_labels(self, texts: List[str], timeout: int = 120) -> List[Sequence[float]]:
            if not texts:
                return []
            resp = self.api.post('/v1/embeddings', {'input': texts}, timeout=timeout)
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

        # ── 3-stage pipelined batch embedding ─────────────────────────────

        def _add_embeddings_for(
            self, *,
            label: str,
            id_prop: str = "id",
            text_prop: str = "label",
            embed_prop: str = "embedding_label",
            batch_size: int = 128,
            concurrency: int = 3,
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
                f"batch_size={batch_size}, ~{n_batches} batches, "
                f"concurrency={concurrency}"
            )

            start_ts = time.time()
            errors: List[Exception] = []

            # ── Stage 1: Reader ────────────────────────────────────────────
            # Streams records from Neo4j, groups them into batches, puts each
            # batch on input_q. Puts `concurrency` sentinels at the end so
            # every embedder thread knows when to stop.
            input_q: Queue = Queue(maxsize=concurrency * 2)

            def _reader():
                try:
                    with self._driver.session(database=self._database) as rs:
                        result = rs.run(fetch_query)
                        buffer_ids: List[str] = []
                        buffer_texts: List[str] = []
                        for rec in result:
                            text = rec["text"]
                            if not text:
                                continue
                            buffer_ids.append(rec["id"])
                            buffer_texts.append(text)
                            if len(buffer_texts) >= batch_size:
                                input_q.put((list(buffer_ids), list(buffer_texts)))
                                buffer_ids.clear()
                                buffer_texts.clear()
                        if buffer_texts:
                            input_q.put((list(buffer_ids), list(buffer_texts)))
                except Exception as exc:
                    logging.error(f"[{log_tag}] Read error: {exc}")
                    errors.append(exc)
                finally:
                    for _ in range(concurrency):   # one sentinel per embedder
                        input_q.put(_SENTINEL)

            # ── Stage 2: Embedders ─────────────────────────────────────────
            # `concurrency` threads pull from input_q independently and call
            # the vLLM API in parallel. Each forwards its sentinel to output_q
            # when it exits so the writer knows how many threads to wait for.
            output_q: Queue = Queue(maxsize=concurrency)

            def _embedder():
                while True:
                    item = input_q.get()
                    if item is _SENTINEL:
                        output_q.put(_SENTINEL)   # forward: signal writer
                        break
                    ids, texts = item
                    try:
                        t0 = time.time()
                        embeddings = self._embed_labels(texts)
                        embed_s = time.time() - t0
                        output_q.put((ids, embeddings, embed_s))
                    except Exception as exc:
                        logging.error(f"[{log_tag}] Embed error: {exc}")
                        errors.append(exc)
                        output_q.put(_SENTINEL)   # still signal writer
                        break

            # ── Stage 3: Writer ────────────────────────────────────────────
            # Single thread drains output_q and writes to Neo4j. Exits after
            # seeing `concurrency` sentinels (one forwarded by each embedder).
            def _writer():
                processed = 0
                batch_idx = 0
                done = 0
                with self._driver.session(database=self._database) as ws:
                    while done < concurrency:
                        item = output_q.get()
                        if item is _SENTINEL:
                            done += 1
                            continue
                        ids, embeddings, embed_s = item
                        try:
                            rows = [
                                {"id": i, "embedding": e}
                                for i, e in zip(ids, embeddings)
                            ]
                            ws.run(write_query, rows=rows)
                            processed += len(rows)
                            batch_idx += 1
                            elapsed = max(time.time() - start_ts, 1e-6)
                            rate = processed / elapsed
                            pct = (processed / total) * 100
                            eta = max(total - processed, 0) / rate if rate > 0 else float("inf")
                            logging.info(
                                f"[{log_tag}] Batch {batch_idx}/{n_batches} | "
                                f"{processed}/{total} ({pct:.1f}%) | "
                                f"embed {embed_s:.1f}s | "
                                f"{rate:.1f} nodes/s | ETA ~{int(eta)}s"
                            )
                        except Exception as exc:
                            logging.error(f"[{log_tag}] Write error: {exc}")
                            errors.append(exc)
                            break

            # ── Launch and join ────────────────────────────────────────────
            reader = Thread(target=_reader, name=f"emb-reader-{label}", daemon=True)
            embedders = [
                Thread(target=_embedder, name=f"emb-worker-{label}-{i}", daemon=True)
                for i in range(concurrency)
            ]
            writer = Thread(target=_writer, name=f"emb-writer-{label}", daemon=True)

            reader.start()
            for e in embedders:
                e.start()
            writer.start()

            reader.join()
            for e in embedders:
                e.join()
            writer.join()

            if errors:
                raise errors[0]

            elapsed = max(time.time() - start_ts, 1e-6)
            logging.info(
                f"[{log_tag}] Done: {total} nodes in {int(elapsed)}s "
                f"({total / elapsed:.1f} nodes/s)"
            )

        # ── Orchestration ──────────────────────────────────────────────────

        def apply_updates(self, batch_size: int = 1024, concurrency: int = 2):
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
                    concurrency=concurrency,
                    log_tag=spec.get("log_tag", spec["label"]),
                )

    return EmbeddingImporter
