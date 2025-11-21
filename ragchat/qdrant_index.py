from typing import List, Sequence
from qdrant_client import QdrantClient
from qdrant_client.http import models
from .config import settings


class QdrantIndex:
    """
    Qdrant wrapper compatible with qdrant-client >= 1.9.0
    """

    def __init__(self, url: str = None, api_key: str | None = None, timeout: float = 20.0):
        url = url or settings.qdrant_url
        api_key = api_key or settings.qdrant_api_key

        self.client = QdrantClient(
            url=url,
            api_key=api_key,
            prefer_grpc=False,
            timeout=timeout,
            check_compatibility=False
        )

    def ensure_collection(self, name: str, dim: int):
        existing = [c.name for c in self.client.get_collections().collections]

        if name not in existing:
            print(f"🆕 Creating collection: {name}")
            self.client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    size=dim,
                    distance=models.Distance.COSINE,
                ),
            )
        else:
            print(f"ℹ️ Collection already exists: {name}")

    def recreate(self, name: str, dim: int):
        print(f"♻️ Recreating collection: {name}")
        self.client.recreate_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=dim,
                distance=models.Distance.COSINE,
            ),
        )

    def _to_vector(self, v: Sequence[float]) -> List[float]:
        return v.tolist() if hasattr(v, "tolist") else list(v)

    def upsert(self, name: str, vectors, payloads, start_id: int):
        points = []
        for offset, (v, payload) in enumerate(zip(vectors, payloads)):
            points.append(
                models.PointStruct(
                    id=start_id + offset,
                    vector=self._to_vector(v),
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=name,
            points=points,
            wait=True
        )

        print(f"✅ Inserted {len(points)} → {name}")

    def search(self, name: str, vector, top_k: int = 5):
        qvec = self._to_vector(vector)

        query = models.NearestQuery(
            nearest=qvec
        )

        response = self.client.query_points(
            collection_name=name,
            query=query,
            limit=top_k,
            with_vectors=False,
            with_payload=True,
        )

        return response.points
