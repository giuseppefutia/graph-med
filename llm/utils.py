from util.api_client import ApiClient
from typing import List, Sequence

class EmbedAPI:
    """Thin client around an embedding endpoint exposed by ApiClient."""

    def __init__(self, api: ApiClient):
        self.api = api

    def embed(self, text: str) -> List[float]:
        """Return a single embedding vector."""
        resp = self.api.post("/embed", {"input": [text]})
        return resp[0]["data"][0]

    def embed_many(self, texts: List[str]) -> List[Sequence[float]]:
        """Return one vector per input text; preserve order."""
        if not texts:
            return []
        resp = self.api.post("/embed", {"input": texts})
        vectors = resp[0]["data"]
        if len(vectors) != len(texts):
            raise RuntimeError(f"Embedding count mismatch ({len(vectors)} != {len(texts)})")
        return vectors