import logging

def icd_chapter_factory(base_importer_cls: str, backend: str):

    class ICDChapterImporter(base_importer_cls):
        def __init__(self):
            super().__init__()
            self.backend = backend
        
        # ──────────────────────────────────────────────────────────────
        # Helpers: CSV parsing
        # ──────────────────────────────────────────────────────────────
        @staticmethod
        def get_csv_size(icd_chapter_file):
            return sum(1 for _ in ICDChapterImporter.get_rows(icd_chapter_file))
        
        @staticmethod
        def get_rows(icd_chapter_file):
            import csv
            with open(icd_chapter_file, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter=";")
                for row in reader:
                    yield {
                        "id": row[0],
                        "chapterName": row[1],
                    }

        # ──────────────────────────────────────────────────────────────
        # Node and relationship operations
        # ──────────────────────────────────────────────────────────────
        def merge_nodes(self, icd_chapter_file) -> str:
            query = """
            UNWIND $batch AS item
            MERGE (d:IcdChapter {id: item.id})
            SET d.chapterName = item.chapterName
            """
            size = self.get_csv_size(icd_chapter_file)
            self.batch_store(query, self.get_rows(icd_chapter_file), size=size)

        def merge_rels(self, icd_file) -> str:
            query= """
            UNWIND $batch AS item
            WITH item
            MATCH (p:IcdChapter {id: item.id})
            MATCH (c:IcdDisease {chapter: item.id})
            MERGE (p)-[:CHAPTER_HAS_DISEASE]->(c)
            """
            size = self.get_csv_size(icd_file)
            self.batch_store(query, self.get_rows(icd_file), size=size)

        # ──────────────────────────────────────────────────────────────
        # Orchestration
        # ──────────────────────────────────────────────────────────────
        def import_data(self, icd_chapter_file):

            logging.info("Loading ICD chapter nodes...")
            self.merge_nodes(icd_chapter_file)

            logging.info("Creating rels between chapters and diseases...")
            self.merge_rels(icd_chapter_file)

    return ICDChapterImporter

if __name__ == '__main__':
    from util.cli_entry import run_backend_importer

    run_backend_importer(
        icd_chapter_factory,
        description="Run ICD Chapter Importer.",
        file_help="Path to the ICD Chapter CSV file",
        default_base_path="./data/ontology/icd10/"
    )
