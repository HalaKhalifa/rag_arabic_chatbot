from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct

class QdrantIndex:
    def __init__(self, url: str, api_key: str | None = None):
        self.client = QdrantClient(url=url, api_key=api_key, prefer_grpc=False, timeout=30.0, check_compatibility=False)

    def ensure_collection(self, name: str, dim: int):
        """
        Create the collection only if it doesn't already exist.
        """
        existing = [c.name for c in self.client.get_collections().collections]
        if name not in existing:
            print(f"🆕 Creating new Qdrant collection: {name}")
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
        else:
            print(f"✅ Using existing collection: {name}")

    def recreate(self, name: str, dim: int):
        """
        Force recreation.
        """
        existing = [c.name for c in self.client.get_collections().collections]
        if name in existing:
            print(f"♻️ Recreating collection: {name}")
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
        return self.client.search(
            collection_name=name,
            query_vector=vector.tolist(),
            limit=top_k,
        )