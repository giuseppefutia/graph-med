# graph-med

Graph-based medical platform that imports medical ontologies (ICD-10, HPO, UMLS) into a Neo4j graph database and provides LLM-powered querying and patient phenotype analysis.

---

## Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
- [Data Pipeline](#data-pipeline)
- [Import Commands](#import-commands)
- [Querying & Analysis](#querying--analysis)
- [Graph Export](#graph-export)
- [Neo4j Graph Schema](#neo4j-graph-schema)
- [Neo4j & APOC Virtualization](#neo4j--apoc-virtualization)
- [Notebooks](#notebooks)

---

## Installation

### Python

Install uv to manage Python versions and the required libraries from the requirements.txt file.

```bash
brew install uv
uv --version # uv 0.10.4 (Homebrew 2026-02-17)
uv python install 3.13
uv python list --only-installed
uv venv --python 3.13 --seed
source venv/bin/activate
uv pip install -r requirements.txt
```

### Neo4j

Code has been tested using an instance on Neo4j Desktop (DB version: 2025.11.2).

Default database: `nodes2026` (configured in `config.ini`).

---

## Configuration

All connection details and API endpoints are stored in `config.ini` at the project root.

```ini
[neo4j]
uri      = bolt://localhost:7687
user     = neo4j
password = password
encrypted = 0
database = nodes2026

[chat-api]
uri = https://<medgemma-host>/v1    # LLM inference endpoint

[embedding-api]
uri = https://<qwen-emb-host>       # Embedding API

[gnn-api]
uri = https://<gnn-host>            # GNN API
```

| Section | Used by | Required for |
|---------|---------|--------------|
| `[neo4j]` | all scripts | always |
| `[chat-api]` | `factory/mapper/icd_hpo_auto.py`, `llm/` | Step 3b |
| `[embedding-api]` | `factory/embedding/` | Step 2 only |
| `[gnn-api]` | reserved | future GNN path |

> **Note:** Once embeddings are stored in Neo4j (Step 2), the `[embedding-api]` is no longer needed at runtime for the mapping steps.

---

## Data Folder Structure

Place your data files in the appropriate subdirectory before running the import commands.

```
data/
├── ontology/
│   ├── icd10/              ← ICD-10 text files
│   │   ├── icd102019syst_codes.txt
│   │   ├── icd102019syst_chapters.txt
│   │   └── icd102019syst_groups.txt
│   ├── umls/               ← UMLS metathesaurus
│   │   └── MRCONSO.RRF
│   └── snomed/             ← (reserved for future use)
└── sample/                 ← Patient data
    └── patient_annotated.csv
```

> **Note:** `hp.owl` (HPO ontology) is downloaded automatically by the HPO importer — no manual placement needed.

---

## Data Pipeline

### HPO Import

The HPO pipeline loads:

- The full HPO ontology (from `hp.owl` via `rdflib-neo4j`)
- Precomputed HPO embeddings

### ICD-10 Import

The ICD-10 pipeline loads:

- Disease codes + hierarchy
- Chapters
- Groups
- Embeddings

### UMLS Concept Mapping

Maps UMLS concepts (from `MRCONSO.RRF`) to:

- ICD-10 Codes
- HPO Terms

### Patient Data

Import patient-generated annotations (CSV format).

---

## Import Commands

Run all commands from the project root with the virtual environment activated.

### 1. Import Structure

```bash
python -m factory.importer.hpo --backend neo4j
python -m factory.importer.icd10_disease --backend neo4j --file icd102019syst_codes.txt
python -m factory.importer.icd10_chapter --backend neo4j --file icd102019syst_chapters.txt
python -m factory.importer.icd10_group --backend neo4j --file icd102019syst_groups.txt
python -m factory.importer.patient_annotation --backend neo4j --file patient_annotated.csv
```

### 2. Compute Embeddings

Requires the embedding API URI to be set in `config.ini`:

```ini
[embedding-api]
uri = https://<your-ngrok-or-host>
```

Run with defaults (tuned for GTE-Qwen2-7B-Instruct on A100):

```bash
python -m factory.embedding.hpo_embedding --backend neo4j
python -m factory.embedding.icd10_embedding --backend neo4j
```

Override batch size and concurrency if needed:

```bash
python -m factory.embedding.hpo_embedding --backend neo4j --batch_size 512 --concurrency 1
```

| Flag | Default | Description |
|------|---------|-------------|
| `--batch_size` | 1024 | Texts per API request — increase for faster GPUs, decrease if you get 422 errors |
| `--concurrency` | 2 | Parallel API requests in flight — reduce to 1 if ngrok throttles |

### 3. Map Ontologies

**Step 3a — UMLS structural mapping** (ICD ↔ HPO via UMLS concepts):

```bash
python -m factory.mapper.icd_hpo_umls --backend neo4j --file MRCONSO.RRF
```

Produces: `(:IcdDisease)-[:UMLS_TO_ICD]->(:UMLS)-[:UMLS_TO_HPO_PHENOTYPE]->(:HpoPhenotype)`

**Step 3b — Embedding + LLM mapping** (vector similarity → LLM disambiguation):

```bash
python -m factory.mapper.icd_hpo_auto --backend neo4j
```

Produces: `(:IcdDisease)-[:ICD_MAPS_TO_HPO_BY_EMBEDDING]->(:HpoPhenotype)`

Requires: `[chat-api]` in `config.ini`. Uses stored `embedding_label` vectors from Step 2 — no embedding API needed at runtime.

---

## Querying & Analysis

Once the graph is built, the LLM agent (`llm/agent.py`) provides natural language access to the knowledge graph. It uses a LangGraph state machine that routes queries through three modes:

| Mode | Trigger | Description |
|------|---------|-------------|
| `stepwise` | Patient ID detected | ICD codes → HPO mapping → ancestor rollup → disease coverage ranking |
| `patient_info` | Keywords like "summarize", "findings", "treatment" | Clinical narrative from APOC data virtualization |
| `text2cypher` | Fallback | LLM-generated Cypher with iterative validation |

### Running the Agent

**Python:**

```python
from llm.agent import run_agent

result = run_agent("Rank diseases by HPO coverage for patientId:'P003'")
```

**CLI:**

```bash
python -m llm.agent
```

The `stepwise` pipeline (`llm/query_factory.py`) is fully deterministic — no LLM in the loop:

1. Fetch patient ICD codes via APOC data virtualization
2. Map ICD → HPO through embedding-based relationships + UMLS bridge
3. Roll up HPO terms to ancestors via `subClassOf` hierarchy
4. Compute disease coverage (% of phenotypes matched) and rank

The LLM is only used at the edges: guardrails (domain check), `text2cypher` fallback, and final answer formatting.

---

## Graph Export

Export the ontology graph to a pickle file for GNN training:

```bash
python -m factory.engine.export_graph --output data/ontology_graph.pkl
```

| Flag | Default | Description |
|------|---------|-------------|
| `--output` | `data/ontology_graph.pkl` | Output file path |
| `--embedding-dim` | 3584 | Embedding dimension |

The export includes hierarchy edges only (`HAS_CHILD`, `GROUP_HAS_DISEASE`, `GROUP_IN_CHAPTER`, `subClassOf`, `HAS_PHENOTYPIC_FEATURE`) and deliberately excludes mapping edges (`ICD_MAPS_TO_HPO_BY_EMBEDDING`, `UMLS_TO_*`) to avoid evaluation bias.

Output: ~45K nodes, ~330K edges with pre-computed embeddings (dim 3584).

---

## Neo4j Graph Schema

### Node Labels

| Label | Count | Description |
|-------|-------|-------------|
| `IcdDisease` | ~12,150 | ICD-10 disease codes |
| `IcdChapter` | 22 | ICD-10 chapters |
| `IcdGroup` | ~400 | ICD-10 groups |
| `HpoPhenotype` | ~19,900 | HPO phenotype terms |
| `HpoDisease` | ~13,000 | HPO disease definitions |
| `Umls` | varies | UMLS concept bridge nodes |

### Relationship Types

```
(:IcdChapter)-[:CHAPTER_HAS_DISEASE]->(:IcdDisease)
(:IcdGroup)-[:GROUP_HAS_DISEASE]->(:IcdDisease)
(:IcdGroup)-[:GROUP_IN_CHAPTER]->(:IcdChapter)
(:IcdDisease)-[:HAS_CHILD]->(:IcdDisease)
(:HpoPhenotype)-[:subClassOf]->(:HpoPhenotype)
(:HpoDisease)-[:HAS_PHENOTYPIC_FEATURE]->(:HpoPhenotype)
(:IcdDisease)-[:ICD_MAPS_TO_HPO_BY_EMBEDDING]->(:HpoPhenotype)
(:Umls)-[:UMLS_TO_ICD]->(:IcdDisease)
(:Umls)-[:UMLS_TO_HPO_PHENOTYPE]->(:HpoPhenotype)
```

> **Note:** `subClassOf` is camelCase (from `rdflib-neo4j`), not `SUBCLASSOF`.

---

## Neo4j & APOC Virtualization

### APOC Extended

APOC Extended is required for data virtualization.

* Go to: https://github.com/neo4j-contrib/neo4j-apoc-procedures/releases
* Download: https://github.com/neo4j-contrib/neo4j-apoc-procedures/releases/download/2025.11.0/apoc-2025.11.0-extended.jar
* In Neo4j Desktop:
    * Make sure your database is stopped
    * Click the three dots (...) next to your database
    * Select Open -> Instance folder -> Plugins
* Copy the apoc-2025.11.0-extended.jar into that plugins folder.
* In neo4j.conf:
    * dbms.security.procedures.unrestricted=apoc.*
    * dbms.security.procedures.allowlist=apoc.*
* In apoc.conf:
    * apoc.import.file.enabled=true
    * apoc.import.file.use_neo4j_config=true
* Start the database again from Neo4j Desktop
* In Neo4j Browser:
    * RETURN apoc.version()
    * SHOW PROCEDURES YIELD name WHERE name STARTS WITH 'apoc.dv' RETURN name

### Testing Data Virtualization

From the system database:

```cypher
CALL apoc.dv.catalog.install(
  "encounter", "nodes2026",
  {
    type: "CSV",
    url: "file:///patient.csv",
    labels: ["Encounter"],
    query: "map.patientId = $patientId",
    desc: "Encounter details based on the patientId."
  }
);
```

Check the following:

```cypher
CALL apoc.dv.catalog.list()
```

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `notebook/01_llm_demo.ipynb` | Full platform walkthrough: connectors (Embedding, LLM, Neo4j, GNN, GRetriever), knowledge graph overview, ontology mapping, GNN-enhanced mapping, patient annotation (NER/NED), GraphRAG agent, and three-way comparison of mapping backends |
