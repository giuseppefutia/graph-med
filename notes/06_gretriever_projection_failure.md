# GRetriever Projection Training — Failure Analysis

## Context

Notebook 05 (`notebook/run/05_gretriever_projection.ipynb`) attempted a projection-based
(distill-then-train) approach to GRetriever, inspired by LLaVA stage 1. The idea: cache
LLM hidden states once, then train the GNN+MLP to match them via MSE — removing the LLM
from the training loop for ~100x speedup.

## Results

| Method | Accuracy |
|---|---|
| Qwen cosine top-1 | 62.5% |
| Text-only MedGemma | 56.5% |
| GRetriever (projection) | **0.0%** |

MSE training converged well (val loss 0.000232, early stopped at epoch 15), but the
model produced 0 correct predictions. Soft tokens actively disrupted the LLM — 0% of
both-wrong cases shared the same prediction as text-only, meaning the tokens injected
completely incoherent information.

## Root Cause: Context Mismatch Between Caching and Inference

### What was cached

For each (ICD, HPO) training pair, the full sequence **including the correct answer**
was run through MedGemma:

```
Input: "ICD: E11 — Type 2 diabetes. Candidates: ... Best matching HPO ID: HP:0000819"
```

Hidden states at the first 4 positions of the last layer were saved as regression targets.

### Why these targets are invalid at inference

Due to self-attention, hidden states at positions 0–3 attend to **all** positions,
including the answer tokens at the end. These states encode "I've already seen the
correct answer is HP:0000819" — information that flows backward through attention.

At inference, the sequence is:

```
[soft_tok_1..4 | "ICD: E11 ... Best matching HPO ID:"]  → generate from here
```

The answer tokens are absent. The GNN produces soft tokens that mimic states shaped by
a context that doesn't exist at inference time. The LLM receives conflicting signals:
the soft tokens "expect" the answer to be in the sequence, but it isn't, corrupting
generation completely.

### Why MSE loss was misleading

MSE measures vector distance in hidden space, not downstream generation quality. The GNN
learned a near-perfect mapping from subgraph to hidden states (low MSE), but those states
only make sense when the full sequence (with answer) is present. Perfect regression to
the wrong target.

## Comparison with Notebook 04 (End-to-End)

Notebook 04 trains GNN+MLP by backpropagating cross-entropy loss through the frozen LLM.
This avoids the context mismatch because:

1. During training, soft tokens are prepended to the prompt **without** the answer
2. The LLM generates tokens, and the loss measures whether it produces the correct HPO
3. The gradient signal directly reflects generation quality

The tradeoff: every training step requires a full LLM forward+backward pass (~2s vs ~0.002s),
limiting training to ~10 epochs on an A100.

## Lessons Learned

1. **Transformer hidden states are context-dependent.** States from position `i` encode
   information from the entire sequence via self-attention. Caching states from one context
   and applying them in a different context produces undefined behavior.

2. **Low regression loss does not imply task performance.** MSE convergence in hidden space
   is necessary but not sufficient. The proxy objective must align with the actual task.

3. **LLaVA-style projection works when contexts match.** In LLaVA, the visual encoder
   processes an image (always present at both train and inference time). There is no
   context mismatch. In our setup, the cached states depended on the answer, which is
   absent at inference.

## Potential Fixes (Untested)

### A. Cache from prompt-only sequences
Cache hidden states from the prompt **without** the answer. But then the states don't
encode which HPO is correct — they're identical regardless of the target. You'd need a
different signal, e.g., contrastive: cache states for correct vs incorrect completions
and train the GNN to produce tokens closer to the correct-completion states.

### B. Cache from an intermediate position
Instead of caching states at positions 0–3 (which attend to the full sequence), cache
the hidden state at the **last prompt position** (just before generation starts). This
state summarizes the prompt and influences the first generated token. The GNN could be
trained to shift this state toward the correct-answer direction.

### C. Use the end-to-end approach with more data
Notebook 04's end-to-end training is sound but data-limited. Combining it with expanded
training data (phenotype.hpoa, as proposed in `notes/04_gretriever_improvements.md`)
may be more productive than fixing the projection approach.
