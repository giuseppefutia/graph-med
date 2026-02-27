# graph-med

# Installation
Here you find the details to install the required libraries and plugins.

## Python

Install uv for manage Python versions and the required libraries from the requirements.txt file.

```bash
brew install uv
uv --version # uv 0.10.4 (Homebrew 2026-02-17)
uv python install 3.13
uv python list --only-installed
uv venv --python 3.13 --seed
source venv/bin/activate
uv pip install -r requirements.txt
```

## Neo4j
Code has been tested using an instance on Neo4j Desktop (DB version: 2025.11.2)

### APOC Extended
APOC Extended is required for data virtualization.

* Go to: https://github.com/neo4j-contrib/neo4j-apoc-procedures/releases
* Download: https://github.com/neo4j-contrib/neo4j-apoc-procedures/releases/download/2025.11.0/apoc-2025.11.0-extended.jar
* In Neo4j Desktop:
    * Make sure your database is stopped
    * Click the three dots (⋯) next to your database
    * Select Open → Instance folder → Plugins
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
    query: "map.PatientID = $patientId",
    desc: "Encounter details based on the patientId."
  }
);
```

Check the following:

```cypher
CALL apoc.dv.catalog.list()
```

# Building

```bash
python -m factory.importer.hpo --backend neo4j
```