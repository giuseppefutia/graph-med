import logging
import re

def icd_group_factory(base_importer_cls: str, backend: str):

    # ──────────────────────────────────────────────────────────────
    # Helper: alphanumeric range generator
    # ──────────────────────────────────────────────────────────────
    def alphanum_range(start: str, end: str):
        """
        Generate an inclusive range like A00..A09 or B35..B64.
        Supports multi-letter prefixes (e.g., 'AA007'..'AA012').
        The letter prefix must match in start and end.
        Maintains zero-padding based on the larger of the two inputs.
        """
        m1 = re.fullmatch(r'([A-Za-z]+)(\d+)', start)
        m2 = re.fullmatch(r'([A-Za-z]+)(\d+)', end)
        if not m1 or not m2:
            raise ValueError("Inputs must be like 'A00', 'B35', 'AA007' (letters + digits).")
        
        p1, n1s = m1.groups()
        p2, n2s = m2.groups()
        if p1 != p2:
            print("Prefixes must match (e.g., both 'B').")
            return []
        
        n1, n2 = int(n1s), int(n2s)
        width = max(len(n1s), len(n2s))  # preserve padding
        step = 1 if n2 >= n1 else -1
        
        return [f"{p1}{i:0{width}d}" for i in range(n1, n2 + step, step)]


    class ICDGroupImporter(base_importer_cls):
        def __init__(self):
            super().__init__()
            self.backend = backend
        
        # ──────────────────────────────────────────────────────────────
        # Helpers: CSV parsing
        # ──────────────────────────────────────────────────────────────
        @staticmethod
        def get_csv_size(icd_group_file):
            return sum(1 for _ in ICDGroupImporter.get_rows(icd_group_file))

        @staticmethod
        def get_rows(icd_group_file):
            import csv
            with open(icd_group_file, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter=";")
                for row in reader:
                    yield {
                        "id": row[2],
                        "name": row[3],
                        "diseaseRange": alphanum_range(row[0], row[1])
                    }

        # ──────────────────────────────────────────────────────────────
        # Node and relationship operations
        # ──────────────────────────────────────────────────────────────
        def merge_nodes(self, icd_group_file) -> str:
            query = """
            UNWIND $batch AS item
            MERGE (d:IcdGroup {id: item.id})
            SET d.groupName = item.name,
                d.diseaseRange = item.diseaseRange
            """
            size = self.get_csv_size(icd_group_file)
            self.batch_store(query, self.get_rows(icd_group_file), size=size)

        def merge_rels(self, icd_group_file) -> str:
            query= """
            UNWIND $batch AS item
            WITH item
            MATCH (g:IcdGroup {id: item.id})
            MATCH (c:IcdChapter {id: item.id})
            MERGE (g)-[:GROUP_IN_CHAPTER]->(c)
            WITH item, g
            UNWIND item.diseaseRange AS diseaseCode
            MATCH (p:IcdDisease {id: diseaseCode})
            MERGE (g)-[:GROUP_HAS_DISEASE]->(p)
            """
            size = self.get_csv_size(icd_group_file)
            self.batch_store(query, self.get_rows(icd_group_file), size=size)

        # ──────────────────────────────────────────────────────────────
        # Orchestration
        # ──────────────────────────────────────────────────────────────
        def import_data(self, icd_group_file):

            logging.info("Loading ICD group nodes...")
            self.merge_nodes(icd_group_file)

            logging.info("Creating rels between groups and chapters/diseases...")
            self.merge_rels(icd_group_file)

    return ICDGroupImporter

if __name__ == '__main__':
    from util.cli_entry import run_backend_importer

    run_backend_importer(
        icd_group_factory,
        description="Run ICD Group Importer.",
        file_help="Path to the ICD Group CSV file",
        default_base_path="./data/ontology/icd10/"
    )
