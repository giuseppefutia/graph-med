"""
HPO Importer — migrated from neosemantics (n10s) to rdflib-neo4j.

MIGRATION SUMMARY
=================
What changed:
  - Removed: n10s server-side plugin dependency (no more n10s.graphconfig.init,
    n10s.graphconfig.set, n10s.rdf.import.fetch).
  - Added: rdflib + rdflib-neo4j for client-side RDF parsing and ingestion.
  - The HANDLE_VOCAB_URI_STRATEGY.IGNORE setting replaces n10s's
    `handleVocabUris: 'IGNORE'` — URIs are stripped to local names.

What stayed the same:
  - All Cypher post-processing queries (labeling, CSV loading, APOC calls).
  - The `n10s_unique_uri` constraint (rdflib-neo4j still uses Resource nodes
    with a `uri` property).
  - APOC Core is still required for apoc.text.replace, apoc.text.regexGroups,
    and apoc.periodic.iterate.

Caveats:
  - n10s `applyNeo4jNaming: True` auto-converted property names to camelCase
    and relationship types to UPPER_SNAKE_CASE. rdflib-neo4j with IGNORE
    strategy strips the namespace prefix and uses the raw local name from the
    URI. In practice, for the HPO ontology, the resulting labels should be
    equivalent, but you should verify after the first import that property
    names and relationship types look correct. If they don't, you can use
    rdflib-neo4j's `custom_mappings` to remap specific predicates.
  - rdflib-neo4j currently supports import only (no RDF export, SHACL
    validation, or inferencing that n10s provided).
  - The hp.owl file (~180MB) is parsed client-side in Python, which is slower
    than n10s's server-side fetch. Expect the ontology loading step to take
    longer (several minutes depending on hardware/network). Batching is
    enabled to keep memory usage reasonable.

Requirements:
  pip install rdflib rdflib-neo4j neo4j
"""

import logging

from neo4j.exceptions import ClientError as Neo4jClientError
from rdflib import Graph as RDFGraph, Namespace
from rdflib_neo4j import Neo4jStoreConfig, Neo4jStore, HANDLE_VOCAB_URI_STRATEGY


def hpo_factory(base_importer_cls: str, backend: str):

    class HPOImporter(base_importer_cls):

        # ── HPO ontology URL ──────────────────────────────────────────
        HPO_OWL_URL = "http://purl.obolibrary.org/obo/hp.owl"

        # ── Prefixes used in the HPO ontology ─────────────────────────
        # Add/adjust prefixes here if rdflib complains about unknown ones
        HPO_PREFIXES = {
            'obo': Namespace('http://purl.obolibrary.org/obo/'),
            'oboInOwl': Namespace('http://www.geneontology.org/formats/oboInOwl#'),
            'owl': Namespace('http://www.w3.org/2002/07/owl#'),
            'rdfs': Namespace('http://www.w3.org/2000/01/rdf-schema#'),
            'rdf': Namespace('http://www.w3.org/1999/02/22-rdf-syntax-ns#'),
            'skos': Namespace('http://www.w3.org/2004/02/skos/core#'),
            'dc': Namespace('http://purl.org/dc/elements/1.1/'),
        }

        def __init__(self):
            super().__init__()
            self.backend = backend
            with self._driver.session() as session:
                session.run(f"CREATE DATABASE {self._database} IF NOT EXISTS")

        # ──────────────────────────────────────────────────────────────
        # Helper: build auth_data dict for rdflib-neo4j from the base
        # class's driver. Adjust this if your base class exposes the
        # connection details differently.
        # ──────────────────────────────────────────────────────────────
        def _get_auth_data(self) -> dict:
            """
            Construct the auth_data dictionary that rdflib-neo4j expects.
            Override or adjust this method to match your base class's
            connection properties.

            Expected base class attributes:
                self._uri       – bolt:// or neo4j:// URI
                self._user      – Neo4j username (usually 'neo4j')
                self._password  – Neo4j password
                self._database  – target database name
            """
            return {
                'uri': self._uri,
                'database': self._database,
                'user': self._user,
                'pwd': self._password,
            }

        # ──────────────────────────────────────────────────────────────
        # 1. Constraints & indexes
        # ──────────────────────────────────────────────────────────────
        def set_constraints(self):
            queries = [
                # Required by rdflib-neo4j — same constraint n10s used
                "CREATE CONSTRAINT n10s_unique_uri IF NOT EXISTS FOR (r:Resource) REQUIRE r.uri IS UNIQUE;",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Resource) REQUIRE (n.id) IS UNIQUE;",
                "CREATE INDEX disease_id IF NOT EXISTS FOR (n:HpoDisease) ON (n.id);",
                "CREATE INDEX phenotype_id IF NOT EXISTS FOR (n:HpoPhenotype) ON (n.id);",
            ]
            with self._driver.session(database=self._database) as session:
                for q in queries:
                    try:
                        session.run(q)
                    except Neo4jClientError as e:
                        if e.code != "Neo.ClientError.Schema.EquivalentSchemaRuleAlreadyExists":
                            raise e

        # ──────────────────────────────────────────────────────────────
        # 2. Load HPO ontology via rdflib-neo4j (replaces n10s import)
        # ──────────────────────────────────────────────────────────────
        def load_HPO_ontology(self):
            """
            Parse hp.owl (RDF/XML) through rdflib and persist every triple
            into Neo4j via the Neo4jStore backend.

            This replaces:
                CALL n10s.graphconfig.init();
                CALL n10s.graphconfig.set({ handleVocabUris: 'IGNORE' });
                CALL n10s.graphconfig.set({ applyNeo4jNaming: True });
                CALL n10s.rdf.import.fetch("http://purl.obolibrary.org/obo/hp.owl", "RDF/XML");
            """
            # Skip if data is already loaded
            with self._driver.session(database=self._database) as session:
                result = session.run("MATCH (n:Resource) RETURN n LIMIT 1")
                if result.data():
                    logging.info("Resource nodes already exist — skipping ontology import.")
                    return

            auth_data = self._get_auth_data()

            config = Neo4jStoreConfig(
                auth_data=auth_data,
                custom_prefixes=self.HPO_PREFIXES,
                handle_vocab_uri_strategy=HANDLE_VOCAB_URI_STRATEGY.IGNORE,
                batching=True,
            )

            neo4j_graph = RDFGraph(store=Neo4jStore(config=config))

            logging.info(
                "Downloading and parsing %s — this may take several minutes...",
                self.HPO_OWL_URL,
            )
            # rdflib will download the URL and parse it as RDF/XML
            neo4j_graph.parse(self.HPO_OWL_URL, format="xml")

            # IMPORTANT: close the store to flush any remaining batched writes
            neo4j_graph.close(commit_pending_transaction=True)
            logging.info("Ontology import complete.")

        # ──────────────────────────────────────────────────────────────
        # 3-7. Post-processing — unchanged Cypher / APOC queries
        # ──────────────────────────────────────────────────────────────
        def label_HPO_entities(self):
            query = """
                    MATCH (n:Resource) 
                    WHERE n.uri STARTS WITH "http://purl.obolibrary.org/obo/HP" 
                    SET n:HpoPhenotype, 
                        n.id = coalesce(n.id, replace(apoc.text.replace(n.uri,'(.*)obo/',''),'_', ':'));
                    """
            with self._driver.session(database=self._database) as session:
                session.run(query)

        def create_disease_entities(self):
            query = """
                    LOAD CSV FROM 'https://github.com/obophenotype/human-phenotype-ontology/releases/latest/download/phenotype.hpoa' AS row 
                    FIELDTERMINATOR '\t'
                    WITH row
                    SKIP 5 
                    MERGE (dis:Resource:HpoDisease {id: row[0]}) 
                    ON CREATE SET dis.label = row[1]; 
                    """
            with self._driver.session(database=self._database) as session:
                session.run(query)

        def create_rels_features_diseases(self):
            query = """
                    LOAD CSV FROM 'https://github.com/obophenotype/human-phenotype-ontology/releases/latest/download/phenotype.hpoa' AS row
                    FIELDTERMINATOR '\t'
                    WITH row
                    SKIP 5
                    MATCH (dis:HpoDisease)
                    WHERE dis.id = row[0]
                    MATCH (phe:HpoPhenotype)
                    WHERE phe.id = row[3]
                    MERGE (dis)-[:HAS_PHENOTYPIC_FEATURE]->(phe)
                    """
            with self._driver.session(database=self._database) as session:
                session.run(query)

        def add_base_properties_to_rels(self):
            query = """
                    LOAD CSV FROM 'https://github.com/obophenotype/human-phenotype-ontology/releases/latest/download/phenotype.hpoa' AS row 
                    FIELDTERMINATOR '\t' 
                    WITH row 
                    SKIP 5 
                    MATCH (dis:HpoDisease)-[rel:HAS_PHENOTYPIC_FEATURE]->(phe:HpoPhenotype)
                    WHERE phe.id = row[3] and dis.id = row[0] 
                    FOREACH(ignoreMe IN CASE WHEN row[4] is not null THEN [1] ELSE [] END| 
                        SET rel.source = row[4]) 
                    FOREACH(ignoreMe IN CASE WHEN row[5] is not null THEN [1] ELSE [] END| 
                        SET rel.evidence = row[5]) 
                    FOREACH(ignoreMe IN CASE WHEN row[6] is not null THEN [1] ELSE [] END| 
                        SET rel.onset = row[6]) 
                    FOREACH(ignoreMe IN CASE WHEN row[7] is not null THEN [1] ELSE [] END| 
                        SET rel.frequency = row[7]) 
                    FOREACH(ignoreMe IN CASE WHEN row[8] is not null THEN [1] ELSE [] END| 
                        SET rel.sex = row[8]) 
                    FOREACH(ignoreMe IN CASE WHEN row[9] is not null THEN [1] ELSE [] END| 
                        SET rel.modifier = row[9]) 
                    FOREACH(ignoreMe IN CASE WHEN row[10] is not null THEN [1] ELSE [] END| 
                        SET rel.aspect = row[10])
                    FOREACH(ignoreMe IN CASE WHEN row[11] is not null THEN [1] ELSE [] END| 
                        SET rel.biocuration = row[11])
                    """
            with self._driver.session(database=self._database) as session:
                session.run(query)

        def enrich_with_descriptive_properties(self):
            query = """
                    CALL apoc.periodic.iterate(
                        "MATCH (dis:HpoDisease)-[rel:HAS_PHENOTYPIC_FEATURE]->(phe:HpoPhenotype) RETURN rel",
                        "SET rel.createdBy = apoc.text.regexGroups(rel.biocuration, 'HPO:(\\\\w+)\\\\[')[0][1],
                        rel.creationDate = apoc.text.regexGroups(rel.biocuration, '\\\\[(\\\\d{4}-\\\\d{2}-\\\\d{2})\\\\]')[0][1],
                        rel.aspectName = 
                        CASE  
                            WHEN rel.aspect = 'P' THEN 'Phenotypic abnormality' 
                            WHEN rel.aspect = 'I' THEN 'Inheritance' 
                        END, 
                        rel.aspectDescription = 
                        CASE 
                            WHEN rel.aspect = 'P' THEN 'Terms with the P aspect are located in the Phenotypic abnormality subontology' 
                            WHEN rel.aspect = 'I' THEN 'Terms with the I aspect are from the Inheritance subontology' 
                        END, 
                        rel.evidenceName = 
                        CASE  
                            WHEN rel.evidence = 'IEA' THEN 'Inferred from electronic annotation' 
                            WHEN rel.evidence = 'PCS' THEN 'Published clinical study' 
                            WHEN rel.evidence = 'TAS' THEN 'Traceable author statement' 
                        END, 
                        rel.evidenceDescription = 
                        CASE 
                            WHEN rel.evidence = 'IEA' THEN 'Annotations extracted by parsing the Clinical Features sections of the Online Mendelian Inheritance in Man resource are assigned the evidence code IEA.' 
                            WHEN rel.evidence = 'PCS' THEN 'PCS is used for information extracted from articles in the medical literature. Generally, annotations of this type will include the pubmed id of the published study in the DB_Reference field.' 
                            WHEN rel.evidence = 'TAS' THEN 'TAS is used for information gleaned from knowledge bases such as OMIM or Orphanet that have derived the information from a published source.' 
                        END, 
                        rel.url = 
                        CASE 
                            WHEN rel.source STARTS WITH 'PMID:' THEN 'https://pubmed.ncbi.nlm.nih.gov/' + apoc.text.replace(rel.source, '(.*)PMID:', '') 
                            WHEN rel.source STARTS WITH 'OMIM:' THEN 'https://omim.org/entry/' + apoc.text.replace(rel.source, '(.*)OMIM:', '') 
                        END",
                    {batchSize: 1000})
                    """
            with self._driver.session(database=self._database) as session:
                session.run(query)

        def remove_unused_node(self):
            query = """
                    CALL apoc.periodic.iterate(
                        "MATCH (n:Resource) RETURN id(n) as node_id",
                        "MATCH (n)
                        WHERE id(n) = node_id AND
                            NOT 'HpoPhenotype' in labels(n) AND
                            NOT 'HpoDisease' in labels(n)
                        DETACH DELETE n",
                        {batchSize:10000})
                    YIELD batches, total return batches, total
                    """
            with self._driver.session(database=self._database) as session:
                session.run(query)

        # ──────────────────────────────────────────────────────────────
        # Orchestration
        # ──────────────────────────────────────────────────────────────
        def apply_updates(self):
            logging.info("Loading constraints and indexes...")
            self.set_constraints()

            # NOTE: check_neo_semantics() and initialize_neo_semantics()
            # are no longer needed — rdflib-neo4j handles everything
            # client-side without requiring any server-side plugin.

            logging.info("Loading HPO ontology via rdflib-neo4j...")
            self.load_HPO_ontology()

            logging.info("Labeling HPO entities...")
            self.label_HPO_entities()

            logging.info("Creating disease entities...")
            self.create_disease_entities()

            logging.info("Creating relationships between features and diseases...")
            self.create_rels_features_diseases()

            logging.info("Adding base properties to relationships...")
            self.add_base_properties_to_rels()

            logging.info("Enriching relationships with descriptive properties...")
            self.enrich_with_descriptive_properties()

            logging.info("Removing unused nodes...")
            self.remove_unused_node()

    return HPOImporter


if __name__ == '__main__':
    from util.cli_entry import run_backend_importer

    run_backend_importer(
        hpo_factory,
        description="Run HPO Importer with selected backend.",
        file_help="No file needed for HPO importer.",
        default_base_path="./data/",
        require_file=False,
    )