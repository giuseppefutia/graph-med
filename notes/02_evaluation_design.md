# GNN Evaluation Design

## Objective

Measure the impact of GNN integration on ICD → HPO mapping quality, compared to the current vector search + LLM baseline. All required data already exists in the project — no external annotation is needed for the first evaluation pass.

---

See [01_gnn_strategy.md](01_gnn_strategy.md) for the full inventory of `previous/` modules and their status.

---

## Ground Truth Construction

### Silver-standard corpus (UMLS bridge)

The UMLS Metathesaurus already in Neo4j provides a free corpus of known ICD → HPO pairs via shared concepts:

```cypher
MATCH (d:IcdDisease)-[:UMLS_TO_ICD]-(:Umls)-[:UMLS_TO_HPO_PHENOTYPE]->(p:HpoPhenotype)
RETURN DISTINCT d.id AS icd_id, d.label AS icd_label,
                p.id AS hpo_id, p.label AS hpo_label
```

This is the held-out test set. Use it to check whether the system's top-ranked HPO candidate matches the UMLS-known correct answer.

### Agreement analysis (mapper vs. UMLS)

Cross-check existing mapper predictions against the UMLS ground truth to establish the baseline accuracy before any GNN work:

```cypher
MATCH (d:IcdDisease)-[r:ICD_MAPS_TO_HPO_BY_EMBEDDING]->(h:HpoPhenotype)
OPTIONAL MATCH (d)-[:UMLS_TO_ICD]-(:Umls)-[:UMLS_TO_HPO_PHENOTYPE]->(h)
RETURN d.id, h.id, r.confidence,
       (count(*) > 0) AS umls_agrees
```

---

## Three-Layer Evaluation

### Layer 1 — Ranking quality

**Question:** Does GNN reranking improve candidate selection before the LLM is involved?

**Method:**
1. For each ICD code in the UMLS ground truth, run `select_candidates_in_batch` → get top-K HPO candidates with vector cosine scores
2. Re-rank the same K candidates with GNN subgraph similarity scores using `GraphSimilarity.rank_by_relevance()` (no-training path) or `GNNEncoder.rank_by_relevance()` (trained path)
3. Record the rank of the correct HPO term under each method

**Metrics:**

| Metric | Definition |
|---|---|
| **MRR@5** | Mean Reciprocal Rank — primary metric. Does the correct answer rank 1st or 5th? |
| **Recall@5** | Does the correct answer appear at all in the top 5? |
| **NDCG@5** | Accounts for partial credit when multiple HPO terms are plausible |

### Layer 2 — Disambiguation accuracy

**Question:** Does GNN soft-token injection help MedGemma choose the correct candidate, given identical candidate sets?

**Method:** Hold the top-5 candidates fixed (same for both conditions). Compare:

| Condition | Retrieval | LLM input |
|---|---|---|
| Baseline | Vector cosine rank | Flat JSON context (current) |
| GNN-augmented | Same candidates | 4 graph soft tokens + text |

**Metric:** `Accuracy@1` — does `best_id` match the UMLS ground truth?

This isolates the LLM soft-token contribution from the reranking contribution.

### Layer 3 — Clinical coherence on patient data

**Question:** Do the GNN-mapped HPO phenotype sets better characterize patient narratives?

**Method:** Using `data/sample/patient_annotated.csv`, for each patient's confirmed ICD codes:
1. Run both pipelines → obtain HPO phenotype sets
2. Compare the resulting phenotype profiles against the clinical narrative text

**Metrics:**

| Metric | Definition |
|---|---|
| **Coverage** | Fraction of ICD codes that received a confident mapping (confidence ≥ 0.7) |
| **Calibration curve** | Plot confidence score vs. actual accuracy (UMLS agreement rate) |
| **ECE** | Expected Calibration Error — quantifies how trustworthy the confidence score is |
| **Clinical coherence** | Qualitative: does MedGemma produce a more coherent patient summary from GNN-derived phenotypes? |

---

## Ablation Study

To disentangle contributions of each GNN component, run four conditions:

| ID | Retrieval | LLM context | Tests |
|---|---|---|---|
| **A** | Vector cosine | None (top-1 pick, no LLM) | Pure retrieval baseline |
| **B** | Vector cosine | Flat JSON (current system) | Current pipeline |
| **C** | GNN reranking | Flat JSON | Reranking-only contribution |
| **D** | GNN reranking | GNN soft tokens (full) | Full GNN pipeline |

Comparing B vs. C isolates reranking quality. Comparing C vs. D isolates the soft-token injection contribution.

---

## Notebook Structure (`02_gnn_test.ipynb`)

Suggested cell organization:

1. **Setup** — Neo4j connection, imports: `GraphSimilarity`, `GNNEncoder`, `Neo4jToPyG`, `ComparisonClient`
2. **Ground truth** — UMLS query → build test DataFrame
3. **Baseline predictions** — run `select_candidates_in_batch` (current pipeline), store vector cosine scores
4. **Path A predictions** — run `GraphSimilarity.rank_by_relevance()` on same candidates, store scores
5. **Path B predictions** — run `GNNEncoder.rank_by_relevance()` on same candidates (requires trained weights), store scores
6. **Layer 1 metrics** — MRR@5, Recall@5, NDCG@5 comparison table across all three methods
7. **Layer 2 metrics** — Accuracy@1 per ablation condition (A, B, C, D)
8. **Calibration** — confidence vs. actual accuracy plot
9. **Attention analysis** — call `GNNEncoder.get_attention_weights()` on a sample ICD code, visualize per-edge attention
10. **Patient analysis** — clinical coherence evaluation on `patient_annotated.csv`

See `previous/GNN_Capabilities_Demo.ipynb` for reference patterns on multi-hop analysis and comparison setup.

---

## Notes on Ground Truth Limitations

- UMLS is known to be incomplete in the ICD ↔ HPO space: some correct mappings are absent from UMLS
- Silver-standard evaluation may undercount true positives (the mapper may be correct even when UMLS has no record)
- A small manually-validated set (even 50–100 pairs) would strengthen the evaluation significantly and is recommended before publication
