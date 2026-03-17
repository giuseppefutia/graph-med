import logging

from neo4j.exceptions import ClientError as Neo4jClientError

def icd_factory(base_importer_cls: str, backend: str):

    class ICDImporter(base_importer_cls):
        def __init__(self):
            super().__init__()
            self.backend = backend
            with self._driver.session() as session:
                session.run(f"CREATE DATABASE {self._database} IF NOT EXISTS")
        
        # ──────────────────────────────────────────────────────────────
        # Helpers: CSV parsing
        # ──────────────────────────────────────────────────────────────
        @staticmethod
        def get_csv_size(icd_file):
            return sum(1 for _ in ICDImporter.get_rows(icd_file))
        
        @staticmethod
        def get_rows(icd_file):
            import csv
            with open(icd_file, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter=";")
                for row in reader:
                    yield {
                        "chapter": row[3],
                        "group": row[4],
                        "code": row[6],
                        "label": row[8],
                        "parentLabel": row[9] if row[9] != "" else None,
                    }

        # ──────────────────────────────────────────────────────────────
        # 1. Constraints & indexes
        # ──────────────────────────────────────────────────────────────
        def set_constraints(self):
            schema = [
                ("constraint", "icd_unique_id",
                "CREATE CONSTRAINT icd_unique_id FOR (d:IcdDisease) REQUIRE d.id IS UNIQUE"),
                ("index", "icd_label",
                "CREATE INDEX icd_label FOR (d:IcdDisease) ON (d.label)"),
                ("index", "icd_chapter",
                "CREATE INDEX icd_chapter FOR (d:IcdDisease) ON (d.chapter)"),
                ("index", "icd_group",
                "CREATE INDEX icd_group FOR (d:IcdDisease) ON (d.group)"),
                ("index", "icd_parentLabel",
                "CREATE INDEX icd_parentLabel FOR (d:IcdDisease) ON (d.parentLabel)"),
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
        # 2. Node and relationship operations
        # ──────────────────────────────────────────────────────────────
        def merge_nodes(self, icd_file) -> str:
            query = """
            UNWIND $batch AS item
            MERGE (d:IcdDisease {id: item.code})
            SET d.label = item.label,
                d.chapter = item.chapter,
                d.group = item.group,
                d.parentLabel = item.parentLabel
            """
            size = self.get_csv_size(icd_file)
            self.batch_store(query, self.get_rows(icd_file), size=size)

        def merge_rels(self, icd_file) -> str:
            query= """
            UNWIND $batch AS item
            WITH item
            WHERE item.parentLabel IS NOT NULL AND item.parentLabel <> "" AND item.parentLabel <> item.label
            MATCH (p:IcdDisease {label: item.parentLabel})
            MATCH (c:IcdDisease {label: item.label})
            MERGE (p)-[:HAS_CHILD]->(c)
            """
            size = self.get_csv_size(icd_file)
            self.batch_store(query, self.get_rows(icd_file), size=size)

        # ──────────────────────────────────────────────────────────────
        # Orchestration
        # ──────────────────────────────────────────────────────────────
        def import_data(self, icd_file):
            logging.info("Creating constraints / indexes...")
            self.set_constraints()

            logging.info("Loading ICD nodes (id + label) with UNWIND $batch ...")
            self.merge_nodes(icd_file)

            logging.info("Creating ICD hierarchy (parent -> child) with UNWIND $batch ...")
            self.merge_rels(icd_file)

    return ICDImporter

if __name__ == '__main__':
    from util.cli_entry import run_backend_importer

    run_backend_importer(
        icd_factory,
        description="Run ICD Importer with selected backend.",
        file_help="Path to the ICD CSV file",
        default_base_path="./data/ontology/icd10/"
    )
