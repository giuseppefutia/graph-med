# GNN Integration Strategy

## Context

The project abstract frames five architectural goals:

1. Agent-driven GraphRAG for regulated medical domains
2. Knowledge graphs that unify ontologies, documents, and reports while preserving data privacy
3. GNNs supporting relevance, similarity scoring, and multi-hop reasoning
4. Separation of structural knowledge from patient-level data for governance and compliance
5. Graph-powered RAG enabling explainable, temporally grounded answers

---

## Current Pipeline (baseline)

In `factory/mapper/icd_hpo_auto.py`, the ICD -> HPO mapping is a three-step chain:

1. **Flat vector search** — embed the ICD label string, run `db.index.vector.queryNodes` against HPO embeddings, return top-K candidates ranked by cosine similarity
2. **Cypher context fetch** — pull 1-2 hop neighborhood for the ICD node (group + chapter) and HPO candidate metadata as flat JSON dicts
3. **LLM disambiguation** — send flat JSON to MedGemma, which outputs `best_id + confidence + rationale`

The graph is used only as a data store. Structural topology (hierarchy, transitivity, ontology relationships) never reaches the model.

---

## Impact Assessment

### Where GNNs genuinely help

**Subgraph-aware candidate reranking** — The current vector search compares label string embeddings. Two ICD codes in the same chapter share structural context that flat embeddings only partially capture. A GAT propagating through the ICD chapter/group hierarchy and HPO ancestor tree would capture this more systematically. Expected improvement: moderate — approximately 3-8% on MRR@5.

**Patient-level GraphRAG retrieval** — This is the strongest case. When a clinical query requires multi-hop traversal (*patient -> ICD codes -> HPO phenotypes -> similar patients*), that path across heterogeneous node types is structurally necessary reasoning that a GAT encodes well. This is where the abstract's claims about "explainable, temporally grounded answers" and "multi-hop reasoning" are most defensible.

### Where impact is weak

**LLM soft-token injection (GRetriever bridge)** — The most questionable component. MedGemma already has deep medical knowledge. The 4 soft tokens are not teaching it new facts — they attempt to nudge attention with a compact structural signal. Without a training loop that jointly optimizes the GAT encoder and MLP bridge, those tokens are essentially noise. **Confirmed empirically**: GRetriever adds only +0.8% over text-only MedGemma (see [04_gretriever_improvements.md](04_gretriever_improvements.md)).

**The baseline is already reasonable** — The flat JSON context in `build_context` already captures group name, chapter name, HPO synonyms, and descriptions. MedGemma reads this natural language metadata well. The GNN adds the most value when structure cannot be adequately expressed in text — for 50-hop paths, or aggregating signals from dozens of neighbors. For a 5-candidate, 2-hop disambiguation task, the marginal gain is smaller.

---

## GRetriever Reference Architecture

The `GRetriever_+_MedGemma.ipynb` notebook implements:

| Component | Detail |
|---|---|
| GNN model | Graph Attention Network (GAT) |
| Layers | 2 x GATConv (256 hidden channels, 4 heads -> 1 head) |
| Node features | 3584-dim Qwen embeddings |
| Pooling | Global mean pooling -> single graph embedding (256-dim) |
| LLM bridge | MLP projecting 256 -> `mlp_tokens=4 x hidden_size=4608` |
| Injection | 4 soft tokens prepended to MedGemma `input_embeds` |
| LLM | `google/medgemma-4b-it` in bfloat16 |

---

## Existing Implementation (`previous/` folder)

| File | Class / Function | Status |
|---|---|---|
| `previous/gnn.py` | `MedicalGNN` (GAT), `SimpleGCN` | Ready |
| `previous/neo4j_to_pyg.py` | `Neo4jToPyG.extract_by_codes()` | Ready |
| `previous/graph_similarity.py` | `GraphSimilarity` | Ready (schema fixes applied) |
| `previous/encoder.py` | `GNNEncoder.rank_by_relevance()`, `get_attention_weights()` | Ready |
| `previous/train_selfsupervised.py` | `SelfSupervisedGNN`, `train_self_supervised()` | Ready |
| `previous/service.py` | `GRetrieverInference` (GNN + MedGemma + FastAPI) | Needs endpoint adaptation for local vLLM |
| `previous/client.py` | `ComparisonClient` (GRetriever vs text-RAG) | Ready |
| `previous/GNN_Capabilities_Demo.ipynb` | Demo: multi-hop, siblings, patient-ontology | Reference notebook |

---

## Integration Points

### 1. Subgraph-Aware Candidate Reranking
**Addresses:** goals 3 (relevance, similarity scoring, multi-hop reasoning)

**Where:** `select_candidates_in_batch` in `icd_hpo_auto.py`

Two implementation paths, ordered by increasing complexity:

**Path A — No training required** (`previous/graph_similarity.py`):
Combines existing Qwen embeddings (already in Neo4j) with shortest-path distance, common ancestors, and Jaccard neighbor similarity. Zero training needed.

**Path B — Trained GAT** (`previous/encoder.py` + `previous/neo4j_to_pyg.py`):
Uses `Neo4jToPyG.extract_by_codes()` to pull subgraphs, runs them through `MedicalGNN`, scores by cosine similarity of pooled graph embeddings. Captures that ICD codes in the same chapter/group cluster together, and HPO terms sharing ancestors cluster together.

### 2. GNN Soft Token Injection into MedGemma
**Addresses:** goals 1 (agent-driven GraphRAG), 5 (explainable answers)

**Where:** `disambiguate_candidates` in `icd_hpo_auto.py`

`previous/service.py` implements this as `GRetrieverInference`. Extracts the combined ICD + top-K HPO subgraph and injects 4 soft graph tokens before the text embeddings. The rationale becomes causally connected to graph attention weights, making it structurally grounded rather than purely textual.

**Status:** empirically tested, marginal gains (+0.8%). Improvement paths documented in [04_gretriever_improvements.md](04_gretriever_improvements.md).

### 3. Explainability via Attention Weights
**Addresses:** goal 5 (explainable answers)

GAT layer 2 attention weights over the ICD-HPO subgraph are interpretable per edge. `previous/encoder.py` exposes `GNNEncoder.get_attention_weights(code)` directly.

### 4. Heterogeneous Graph for Knowledge Unification
**Addresses:** goal 2 (unify ontologies, documents, reports)

The same GAT encoder can operate over a unified subgraph including ontology nodes (ICD chapters/groups, HPO ancestors), document nodes (clinical notes), and patient nodes. Different edge types are handled via heterogeneous message passing.

### 5. Governance Boundary via Separate GNN Layers
**Addresses:** goal 4 (governance and compliance)

| Layer | Data | GNN role |
|---|---|---|
| Ontology graph | ICD + HPO + cross-ontology edges | GAT trained offline; no PII |
| Patient graph | Clinical notes, reports, annotations | Separate GAT; access-controlled |
| Fusion | Agent combines both at query time | Graph attention over query-specific subgraph |

---

## Complexity vs. Benefit

| Component | Engineering cost | Expected gain | Verdict |
|---|---|---|---|
| No-training reranking (`GraphSimilarity`) | **Low** | Moderate (MRR@5 +3-8%) | Start here |
| Trained GAT reranking (`GNNEncoder`) | **Low** — run `train_selfsupervised.py` | Moderate to good | Second step |
| GNN soft tokens (`GRetrieverInference`) | **Medium** | Low without joint fine-tuning | Defer (see improvement paths) |
| Patient-level GraphRAG GNN | **Medium** | High for multi-hop queries | Highest priority for abstract |
| Attention-based explainability | **Low** — already implemented | High for regulatory compliance | Include from the start |

---

## Recommended Sequence

1. **GraphSimilarity reranking** — call `GraphSimilarity.rank_by_relevance()` inside `select_candidates_in_batch`. Run Layer 1 ranking evaluation. Reveals whether structural signals help before committing to training.
2. **Pretrain GAT** — run self-supervised training loop on ICD + HPO subgraph (link prediction + feature reconstruction). Swap in `GNNEncoder` and compare MRR@5.
3. **Patient-level GraphRAG** — the GNN's strongest contribution. `Neo4jToPyG.extract_by_patient()` is the entry point.
4. **Attention weights for explainability** — include from Step 1 onward. Edge-level explanations valuable for regulatory compliance.
5. **Collect validated set** — 50-100 clinician-validated ICD -> HPO pairs before publication.

---

## Summary

| Abstract claim | GNN fit | Confidence |
|---|---|---|
| Relevance and similarity scoring | Moderate — reranking improves over flat cosine | Medium |
| Multi-hop reasoning | Strong — matters most in patient GraphRAG | High |
| Explainable answers | Strong — attention weights give edge-level explanation | High |
| Unify ontologies + documents | Moderate — requires heterogeneous training | Medium |
| Governance separation | Strong — architectural boundary naturally enforced | High |
