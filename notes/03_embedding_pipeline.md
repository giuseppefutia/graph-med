# Embedding Pipeline — Design Notes

## Problem

Embedding large node sets (HPO phenotypes, ICD diseases, etc.) involves two
sequential steps per batch:

1. **Embed** — POST a list of texts to a remote GPU server (vLLM via ngrok),
   wait for the response. This is slow: a network round-trip + GPU inference,
   typically 2–10 seconds per batch of 128.
2. **Write** — `UNWIND $rows … SET n.embedding = row.embedding` on a local
   Neo4j instance. This is fast: usually < 1 second per batch.

The naive implementation is purely serial:

```
[embed batch 1] → [write batch 1] → [embed batch 2] → [write batch 2] → …
      3s                1s                3s                1s
total per batch: 4s
```

The write sits idle during the embed, and the embed sits idle during the write.

---

## Solution: producer–consumer pipeline

Separate the two roles into two threads connected by a bounded queue.

```
Main thread (producer)          Writer thread (consumer)
──────────────────────          ────────────────────────
fetch records from Neo4j
buffer 128 records
[embed batch 1]  ───put──▶  queue  ◀──get── [write batch 1]
[embed batch 2]  ───put──▶  queue  ◀──get── [write batch 2]
…
put _SENTINEL    ───────▶  queue           (thread exits)
```

While the writer is writing batch N, the main thread is already computing
embeddings for batch N+1. The critical path becomes:

```
max(embed_time, write_time) per batch   ≈  embed_time   (since embed >> write)
```

Wall-clock improvement ≈ `write_time / (embed_time + write_time) × 100%`.
For embed=3s and write=1s this is ~25%.

---

## Key primitives

### `queue.Queue(maxsize=1)`

Python's `queue.Queue` is thread-safe. `maxsize=1` is the key design choice:

- `queue.put(item)` **blocks** when the queue is already full (has 1 item).
- `queue.get()` **blocks** when the queue is empty.

This gives natural back-pressure: the producer (main thread) cannot race too
far ahead of the consumer (writer). The moment the writer picks up batch N,
the queue has a free slot and the producer can enqueue batch N+1.

```python
queue: Queue = Queue(maxsize=1)
```

If `maxsize` were 0 (unbounded), the producer would embed all batches before
any writes happened — wasting memory and losing the overlap benefit.

### Sentinel value

A unique object used to signal "no more data" to the writer thread:

```python
_SENTINEL = object()   # module-level singleton

# producer side (in finally block, always runs):
queue.put(_SENTINEL)

# consumer side:
item = queue.get()
if item is _SENTINEL:
    break
```

Using `object()` (identity comparison with `is`) instead of `None` avoids
ambiguity if `None` were ever a valid payload.

### Two separate Neo4j sessions

The Neo4j Python driver is thread-safe, but **sessions are not**. Each thread
must own its own session:

```python
# Main thread: read session (streams fetch results)
with self._driver.session(database=self._database) as rs:
    result = rs.run(fetch_query)
    …

# Writer thread: write session (runs UNWIND SET queries)
def _writer():
    with self._driver.session(database=self._database) as ws:
        while True:
            item = queue.get()
            …
            ws.run(write_query, rows=item)
```

Never share a session across threads.

### Error propagation

The writer thread cannot raise directly into the main thread. The pattern
used here is a shared list:

```python
errors: List[Exception] = []

def _writer():
    …
    except Exception as exc:
        logging.error(f"Write error: {exc}")
        errors.append(exc)   # visible to main thread
        break

# After writer.join(), main thread checks:
if errors:
    raise errors[0]
```

The main thread also checks `errors` before every embed call to abort early
if the writer has already failed.

### `finally` guarantees the sentinel

No matter what happens (exception in embed, keyboard interrupt, etc.), the
sentinel must be put on the queue so the writer thread can exit and be joined:

```python
try:
    …  # embed loop
finally:
    queue.put(_SENTINEL)   # always runs
    writer.join()          # wait for writer to finish
```

Without this, the writer thread would block forever on `queue.get()` and the
process would hang.

---

## Complete skeleton (minimal reproducible example)

```python
import logging
from queue import Queue
from threading import Thread
from typing import List

_SENTINEL = object()

def pipeline(items: List[str], batch_size: int = 128):
    """
    Skeleton of the embed-write pipeline.
    Replace `embed_batch` and `write_batch` with your real implementations.
    """

    def embed_batch(texts: List[str]) -> List[object]:
        # slow: remote HTTP call
        ...

    def write_batch(rows: List[dict]):
        # fast: local DB write
        ...

    queue: Queue = Queue(maxsize=1)
    errors: List[Exception] = []

    def _writer():
        while True:
            item = queue.get()
            if item is _SENTINEL:
                break
            try:
                write_batch(item)
            except Exception as exc:
                logging.error(f"Write error: {exc}")
                errors.append(exc)
                break

    writer = Thread(target=_writer, daemon=True)
    writer.start()

    try:
        buffer = []
        for item in items:
            if errors:
                break
            buffer.append(item)
            if len(buffer) >= batch_size:
                embeddings = embed_batch(buffer)          # blocks (slow)
                queue.put(embeddings)                     # hand off to writer
                buffer.clear()
        if not errors and buffer:
            embeddings = embed_batch(buffer)
            queue.put(embeddings)
    finally:
        queue.put(_SENTINEL)
        writer.join()

    if errors:
        raise errors[0]
```

---

## When does this pattern NOT help?

- **Write is the bottleneck** (slower than embed): the main thread will block
  on `queue.put()` waiting for the writer. No overlap is gained.
- **Single batch total**: no pipelining benefit for a one-shot job.
- **Side effects require ordering**: if writes must happen in strict order and
  you need confirmation before the next embed, you'd need a more complex
  handshake.

---

## Concurrent embedding — 3-stage pipeline (implemented)

The 2-stage pipeline still sends one embed request at a time. vLLM accepts
concurrent requests and schedules them together on the GPU. The 3-stage
pipeline adds N embedder threads so N requests are always in flight:

```
[reader] ──▶ input_q ──▶ [embedder-0] ──▶ output_q ──▶ [writer]
                      ──▶ [embedder-1] ──▶
                      ──▶ [embedder-2] ──▶
```

### Stage 1 — Reader thread

Streams records from Neo4j, groups them into batches, puts each batch on
`input_q`. At the end, puts **one sentinel per embedder** so each embedder
thread knows independently when to stop:

```python
input_q: Queue = Queue(maxsize=concurrency * 2)

def _reader():
    try:
        # … stream Neo4j, buffer into batches, put on input_q …
    finally:
        for _ in range(concurrency):   # one sentinel per embedder
            input_q.put(_SENTINEL)
```

`maxsize=concurrency * 2` lets the reader run a little ahead without blocking,
while still bounding memory use.

### Stage 2 — Embedder threads (N concurrent)

Each embedder loops independently: pull a batch from `input_q`, call the vLLM
API, push the result to `output_q`. When it receives its sentinel it
**forwards** a sentinel to `output_q` before exiting, signalling the writer:

```python
output_q: Queue = Queue(maxsize=concurrency)

def _embedder():
    while True:
        item = input_q.get()
        if item is _SENTINEL:
            output_q.put(_SENTINEL)   # forward: one signal per embedder
            break
        ids, texts = item
        embeddings = embed_batch(texts)       # concurrent HTTP call to vLLM
        output_q.put((ids, embeddings))
```

Because N threads each call `embed_batch` independently, N requests are in
flight at the same time. vLLM batches them on the GPU.

### Stage 3 — Writer thread

Drains `output_q` and writes to Neo4j. Exits after seeing exactly `concurrency`
sentinels (one forwarded by each embedder):

```python
def _writer():
    done = 0
    while done < concurrency:
        item = output_q.get()
        if item is _SENTINEL:
            done += 1
            continue
        ids, embeddings = item
        write_batch(ids, embeddings)
```

The sentinel counting is the key: the writer does not exit until every embedder
has finished, so no results are lost.

### Launch and join

```python
reader = Thread(target=_reader, daemon=True)
embedders = [Thread(target=_embedder, daemon=True) for _ in range(concurrency)]
writer = Thread(target=_writer, daemon=True)

reader.start()
for e in embedders: e.start()
writer.start()

reader.join()
for e in embedders: e.join()
writer.join()
```

`daemon=True` ensures threads do not block process exit on unexpected errors.
Joining in this order (reader → embedders → writer) guarantees the writer
always finishes after all results have been enqueued.

### Choosing `concurrency`

| Value | Effect |
|-------|--------|
| 1 | Single embedder — functionally identical to the 2-stage pipeline |
| 2–4 | Good starting point; keeps GPU busy without overwhelming the server |
| >4 | Diminishing returns; may saturate ngrok or vLLM's request queue |

Start with `concurrency=3` (the default). The `embed Xs` field in the logs
tells you whether embed time is still the bottleneck.

---

---

## Hardware & model tuning — GTE-Qwen2-7B-Instruct on A100

This section explains how to choose `batch_size` and `concurrency` for the
specific setup used in this project.

### Server configuration

```python
MODEL               = 'Alibaba-NLP/gte-Qwen2-7B-instruct'
VLLM_HOST           = '0.0.0.0'
VLLM_PORT           = 8000
MAX_MODEL_LEN       = 32768   # max tokens per sequence
TENSOR_PARALLEL_SIZE = 1      # single GPU, no splitting
```

The model is served on a **single A100 40 GB** via vLLM, exposed to the
outside world through an **ngrok tunnel**.

---

### What `MAX_MODEL_LEN` means (and what it does NOT mean)

`MAX_MODEL_LEN = 32768` sets the **maximum number of tokens in a single
input sequence** — not a limit on the batch. A medical label like
`"Abnormality of the cardiovascular system"` tokenises to roughly 6–10
tokens, which is < 0.03 % of 32 768.

The practical consequence is that vLLM pre-allocates KV-cache blocks
dimensioned for sequences up to 32 k tokens. This costs some GPU memory
upfront, even though our actual sequences are tiny. It does **not** cap how
many sequences we can pack into a batch.

> **KV cache (Key-Value cache)**: during a Transformer forward pass, each
> attention layer computes "key" and "value" tensors for every token. For
> generation tasks these are cached across decoding steps. For **embedding**
> (a single forward pass with no decoding), the KV cache is less critical,
> but vLLM still reserves the memory blocks to be safe.

---

### GPU memory budget

| Component | Estimate |
|-----------|----------|
| Model weights (7 B × fp16) | ~14 GB |
| KV-cache pre-allocation (32 k max_len) | ~8 GB |
| **Free for activations** | **~18 GB** |

Activation memory for one forward pass:

```
batch_size × avg_seq_len × hidden_size × n_layers × 2 bytes
   1024    ×     15      ×    3584     ×    28     ×   2   ≈  4.5 GB
```

With 18 GB free, `batch_size=1024` uses only ~25 % of the available headroom —
well within budget.

---

### Why `batch_size` matters more than `concurrency` here

Each embed request is a single HTTP POST carrying `batch_size` texts. The
GPU processes the whole batch in **one forward pass**.

```
batch_size=128   →  ~120 requests for 15 000 HPO phenotypes
batch_size=1024  →  ~15  requests for 15 000 HPO phenotypes
```

Fewer, larger requests mean:
- Less ngrok round-trip overhead (each tunnel traversal costs ~50–200 ms).
- More tokens per GPU forward pass → better GPU utilisation.
- Shorter total wall-clock time even before considering concurrency.

This is the highest-leverage knob for this setup.

---

### Why `concurrency` is still useful (but limited here)

`TENSOR_PARALLEL_SIZE = 1` means **one GPU processes all requests
sequentially**. vLLM's scheduler queues concurrent requests and runs them
one after another (or merges them into a single batch if they arrive closely
enough together).

With short medical labels and an A100, a single forward pass of 512 texts
likely completes in < 1 second. The dominant latency is then the **ngrok
round-trip** (~50–200 ms per request). Having 2 embedder threads in flight
hides that latency:

```
embedder-0  waiting for response     (ngrok latency ~150 ms)
embedder-1  sending next request  ←  this is what concurrency buys
```

Going beyond `concurrency=2` offers diminishing returns here because:
1. The GPU is already fast enough that the queue clears quickly.
2. ngrok free tier throttles if too many simultaneous connections are opened.

---

### Recommended settings for this setup

```python
updater.apply_updates(batch_size=1024, concurrency=2)
```

| Parameter | Value | Reason |
|-----------|-------|--------|
| `batch_size` | 1024 | ~4.5 GB activations, 25 % of 18 GB free — safe and minimises round trips |
| `concurrency` | 2 | Hides one ngrok round-trip; single GPU makes more useless |

With 1024 vs 512:
```
batch_size=512   →  ~30  requests for 15 000 HPO phenotypes
batch_size=1024  →  ~15  requests for 15 000 HPO phenotypes
```
Half the ngrok round-trips for free. Memory is not the constraint here.

If you see ngrok connection errors, reduce `concurrency` to 1.
If vLLM returns a 422 (payload too large), drop `batch_size` to 512.

---

## Files in this project

| File | Role |
|------|------|
| [factory/embedding/base_embedding.py](../factory/embedding/base_embedding.py) | Full implementation of the pipeline |
| [factory/embedding/hpo_embedding.py](../factory/embedding/hpo_embedding.py) | HPO node specs only — delegates to base |
| [factory/embedding/icd10_embedding.py](../factory/embedding/icd10_embedding.py) | ICD node specs only — delegates to base |
