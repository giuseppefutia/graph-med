# GRetriever Improvement Paths

## Current Results (Notebook 04)

| Method | Accuracy |
|---|---|
| Qwen cosine top-1 | 62.5% |
| Text-only MedGemma | 58.6% |
| GRetriever (GNN + MedGemma) | 59.4% |

**Key findings:**
- Soft tokens add +0.8% over text-only (44 wins vs 41 losses — near parity)
- Cosine top-1 remains the strongest method overall
- GRetriever's primary win pattern: *specificity correction* — picks the correct specific HPO child when text-only defaults to a broader parent (31 coherent wins out of 44)
- GNN learned a "prefer rank 1" bias (71% win rate at cosine rank 1, but hurts at rank 2+)
- On hard disambiguation cases (142 where cosine top-1 is wrong): text-only corrects 34, GRetriever only 13
- 51% of both-wrong cases have identical predictions — soft tokens not changing LLM behavior

## Root Cause Analysis

1. **Data scarcity**: 1361 UMLS pairs for training. The GNN+MLP bridge has ~2.2M parameters learning from <1000 effective training examples after deduplication
2. **Frozen LLM**: MedGemma cannot adapt to soft token semantics — it treats them as noise unless they happen to land in a useful region of embedding space
3. **Weak structural signal**: The ICD↔HPO graph has no direct cross-ontology edges except via UMLS bridges. The 2-hop subgraphs mostly contain intra-ontology hierarchy, which doesn't help disambiguation

## Improvement Directions (Ranked by Expected Impact)

### 1. More Training Data (HIGH impact, LOW effort)

**phenotype.hpoa** — HPO disease-to-phenotype annotations
- ~180k disease→phenotype annotations from OMIM, Orphanet, DECIPHER
- Maps OMIM disease IDs to HPO phenotypes
- OMIM↔ICD crosswalk available via UMLS (already in our graph)
- **Estimated yield**: 5-10x more ICD→HPO training pairs
- Download: https://hpo.jax.org/data/annotations

**OMIM morbidmap** — OMIM disease↔gene↔phenotype
- Provides indirect ICD→HPO paths via shared genes
- Useful for constructing negative examples and harder training pairs

**MONDO crosswalks** — MONDO disease ontology
- Maps between ICD-10, OMIM, Orphanet, SNOMED
- Provides validated disease equivalence classes
- Can generate high-confidence ICD→HPO pairs through transitive mappings

**Synthetic hard negatives**
- For each ICD code, pick HPO candidates that are semantically close but wrong (sibling HPO terms)
- Forces the model to learn fine-grained distinctions rather than just "prefer rank 1"

### 2. LoRA on MedGemma (HIGH impact, MEDIUM effort)

The frozen LLM cannot learn to "read" soft tokens. Adding LoRA adapters lets MedGemma learn to interpret graph-structural context.

- **Target layers**: attention Q/V projections in last 4-8 layers
- **Rank**: r=8-16 (adds ~2-5M trainable params)
- **Memory**: LoRA adds ~1-2 GB on top of current ~15 GB footprint — still fits A100
- **Risk**: Overfitting with current data size. Combine with data expansion (path 1) for best results
- **Implementation**: `peft` library, 10-line change to model loading

### 3. More Soft Tokens (LOW impact, LOW effort)

Current: 4 soft tokens. The graph embedding may need more representational capacity.

- Try 8, 16, 32 tokens
- Diminishing returns expected — the bottleneck is likely signal quality, not capacity
- Quick experiment: sweep NUM_TOKENS in [4, 8, 16] and compare val loss

### 4. Different GNN Architecture (LOW impact, MEDIUM effort)

Current: 2-layer GAT with 4 heads, 256 hidden channels.

- **GIN (Graph Isomorphism Network)**: better at distinguishing graph structures
- **GraphSAGE**: inductive, may generalize better to unseen subgraphs
- **Deeper GAT**: 3-4 layers with residual connections
- **Likely outcome**: marginal differences. The GNN architecture is not the bottleneck — the training signal is

### 5. Subgraph Encoding Improvements (MEDIUM impact, MEDIUM effort)

- **Edge-type awareness**: currently edges are untyped. Encoding HAS_CHILD vs UMLS_TO_* as edge features could help
- **Positional encoding**: add random walk positional encodings to distinguish node roles in the subgraph
- **Larger subgraphs**: increase max_nodes from 64 to 128, use 3-hop instead of 2-hop

### 6. Contrastive Pre-training of GNN (MEDIUM impact, HIGH effort)

Before the discriminative task, pre-train the GNN with contrastive learning on the full 32k-node graph:
- Positive pairs: nodes connected by UMLS bridges
- Negative pairs: random ICD↔HPO pairs
- Then fine-tune on the discriminative GRetriever task
- This could give the GNN a better initialization, but notebook 03 showed contrastive approaches struggled

## Recommended Next Steps

1. **Immediate**: Extend training data with phenotype.hpoa (highest ROI)
2. **If data expansion works**: Add LoRA to MedGemma
3. **If still insufficient**: Combine both + sweep soft token count
4. **Architecture changes**: only after data/LoRA have been explored
