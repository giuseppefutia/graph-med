from itertools import islice
from typing import Iterable

from tqdm import tqdm

from database.neo4j_db import Neo4jGraphDB


class Neo4jBaseImporter(Neo4jGraphDB):
    def __init__(self):
        super().__init__()
        self.batch_size = 1000

    def batch_store(self, query: str, generator: Iterable, size: int = None, **kwargs):
        def batched(iterable, n):
            it = iter(iterable)
            while batch := list(islice(it, n)):
                yield batch

        try:
            with self._driver.session(database=self._database) as session:
                batches = batched(generator, self.batch_size)

                if size:
                    total_batches = (size + self.batch_size - 1) // self.batch_size
                    batches = tqdm(batches, total=total_batches, desc="Loading data into Neo4j...")
                else:
                    batches = tqdm(batches, desc="Loading data into Neo4j...")

                for batch in batches:
                    if not batch:
                        continue

                    session.run(query, {"batch": batch})

        except Exception as e:
            print(f"Batch insert failed: {e}")
    
    def create_indices(self, indices):
        for index in indices:
            with self._driver.session(database=self._database) as session:
                session.run(index)