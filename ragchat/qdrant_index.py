from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct

class QdrantIndex:
    def __init__(self, url: str, api_key: str | None = None):
        self.client = QdrantClient(url=url, api_key=api_key)

    def recreate(self, name: str, dim: int):
        # drop if exists, then create fresh
        existing = [c.name for c in self.client.get_collections().collections]
        if name in existing:
            self.client.delete_collection(name)
        self.client.recreate_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    def upsert(self, name: str, vectors, payloads: List[Dict[str, Any]]):
        points = [
            PointStruct(id=i, vector=v.tolist(), payload=payloads[i])
            for i, v in enumerate(vectors)
        ]
        self.client.upsert(collection_name=name, points=points)

    def search(self, name: str, vector, top_k: int = 5):
        """
        Search the Qdrant collection for the closest vectors.
        """
        return self.client.search(
            collection_name=name,
            query_vector=vector.tolist(),
            limit=top_k,
        )
