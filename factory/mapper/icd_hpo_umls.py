import logging

from neo4j.exceptions import ClientError as Neo4jClientError

def umls_map_factory(base_importer_cls: str, backend: str):

    class UMLSMapImporter(base_importer_cls):
        def __init__(self):
            super().__init__()
            self.backend = backend
        
        # ──────────────────────────────────────────────────────────────
        # 1. Constraints & indexes
        # ──────────────────────────────────────────────────────────────
        def set_constraints(self):
            schema = [
                ("constraint", "umls_unique_id",
                 "CREATE CONSTRAINT umls_unique_id FOR (u:Umls) REQUIRE u.id IS UNIQUE"),
            ]

            with self._driver.session(database=self._database) as s:
                exists = {
                    "constraint": {r["name"] for r in s.run("SHOW CONSTRAINTS YIELD name")},
                    "index": {r["name"] for r in s.run("SHOW INDEXES YIELD name")},
                }
                for kind, name, create_cypher in schema:
                    if name in exists[kind]:
                        continue
                    try:
                        s.run(create_cypher)
                    except Neo4jClientError as e:
                        if getattr(e, "code", None) not in {
                            "Neo.ClientError.Schema.ConstraintAlreadyExists",
                            "Neo.ClientError.Schema.IndexAlreadyExists",
                            "Neo.ClientError.Schema.EquivalentSchemaRuleAlreadyExists",
                        }:
                            raise

        # ──────────────────────────────────────────────────────────────
        # Helpers: CSV parsing
        # ──────────────────────────────────────────────────────────────
        @staticmethod
        def get_csv_size(umls_map_file):
            return sum(1 for _ in UMLSMapImporter.get_rows(umls_map_file))

        @staticmethod
        def get_rows(umls_map_file):
            import csv, sys

            _max = sys.maxsize
            while True:
                try:
                    csv.field_size_limit(_max)
                    break
                except OverflowError:
                    _max //= 10

            with open(umls_map_file, "r") as in_file:
                reader = csv.reader(in_file, delimiter="|")
                for row in reader:
                    yield {
                        "id": row[0],
                        "type_source": row[11],
                        "code": row[13]
                    }
        
        # ──────────────────────────────────────────────────────────────
        # Mapping operations
        # ──────────────────────────────────────────────────────────────
        def map_to_icd(self, umls_file):
            icd_query = """
            UNWIND $batch as item
            MATCH (icd:IcdDisease)
            WHERE icd.id = item.code
            WITH item, icd
            MERGE (umls:Umls {id: item.id})
            MERGE (umls)-[:UMLS_TO_ICD]->(icd)
            SET icd.umls_ids = CASE 
                    WHEN item.id in icd.umls_ids THEN icd.umls_ids
                    ELSE coalesce(icd.umls_ids,[]) + item.id END
            """
            size = self.get_csv_size(umls_file)
            self.batch_store(icd_query, self.get_rows(umls_file), size=size)
        
        def map_to_hpo_phen(self, umls_file):
            phen_query = """
            UNWIND $batch as item
            MATCH (hpo:HpoPhenotype)
            WHERE hpo.id = item.code
            WITH item, hpo
            MERGE (umls:Umls {id: item.id})
            MERGE (umls)-[:UMLS_TO_HPO_PHENOTYPE]->(hpo)
            SET hpo.umls_ids = CASE 
                    WHEN item.id in hpo.umls_ids THEN hpo.umls_ids
                    ELSE coalesce(hpo.umls_ids,[]) + item.id END
            """
            size = self.get_csv_size(umls_file)
            self.batch_store(phen_query, self.get_rows(umls_file), size=size)

        def map_to_hpo_disease(self, umls_file):
            disease_query = """
            UNWIND $batch as item
            MATCH (dis:HpoDisease)
            WHERE dis.id = item.type_source + ':' + item.code
            WITH item, dis
            MERGE (umls:Umls {id: item.id})
            MERGE (umls)-[:UMLS_TO_HPO_DISEASE]->(dis)
            SET dis.umls_ids = CASE 
                    WHEN item.id in dis.umls_ids THEN dis.umls_ids
                    ELSE coalesce(dis.umls_ids,[]) + item.id END
            """
            size = self.get_csv_size(umls_file)
            self.batch_store(disease_query, self.get_rows(umls_file), size=size)

        # ──────────────────────────────────────────────────────────────
        # Orchestration
        # ──────────────────────────────────────────────────────────────
        def import_data(self, umls_map_file):

            logging.info("Creating constraints / indexes...")
            self.set_constraints()

            logging.info("Mapping UMLS to ICD...")
            self.map_to_icd(umls_map_file)

            logging.info("Mapping UMLS to HPO Phenotypes...")
            self.map_to_hpo_phen(umls_map_file)

            logging.info("Mapping UMLS to HPO Diseases...")
            self.map_to_hpo_disease(umls_map_file)

    return UMLSMapImporter

if __name__ == '__main__':
    from util.cli_entry import run_backend_importer

    run_backend_importer(
        umls_map_factory,
        description="Run UMLS Map Importer.",
        file_help="Path to the UMLS Map CSV file",
        default_base_path="./data/ontology/umls/"
    )
