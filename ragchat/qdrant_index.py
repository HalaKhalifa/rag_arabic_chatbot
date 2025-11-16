from typing import List, Dict, Any, Sequence
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct


class QdrantIndex:
    def __init__(self, url: str, api_key: str | None = None, timeout: float = 20.0):
        """
        Simple wrapper around QdrantClient for collection management,
        upsert, and search.
        """
        self.client = QdrantClient(url=url, api_key=api_key, prefer_grpc=True, timeout=timeout, check_compatibility=False)

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
            print(f"ℹ️ Qdrant collection already exists: {name}")

    def recreate(self, name: str, dim: int):
        """
        Force recreation.
        """
        print(f"♻️ Recreating collection: {name}")
        self.client.recreate_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    def _to_vector(self, v: Sequence[float]) -> List[float]:
        """
        Safely convert numpy arrays or lists/tuples to a plain Python list.
        """
        if hasattr(v, "tolist"):
            return v.tolist()
        return list(v)

    def upsert(self, name: str, vectors, payloads, start_id: int):
        """
        Upsert a batch of points into Qdrant with globally unique IDs.
        - vectors: iterable of embedding vectors (list[list[float]] or np.ndarray)
        - payloads: same length as vectors, each is a dict
        """
        points: List[PointStruct] = []

        for offset, (v, payload) in enumerate(zip(vectors, payloads)):
            point_id = start_id + offset

            points.append(
                PointStruct(
                    id=point_id,
                    vector=self._to_vector(v),
                    payload=payload,
                )
            )

        self.client.upsert(collection_name=name, points=points)
        print(f"✅ Upserted {len(points)} points into '{name}' (IDs {start_id}–{start_id + len(points) - 1})")

    def search(self, name: str, vector, top_k: int = 5):
        """
        Search top_k nearest neighbors for a given query vector.
        """
        query_vector = self._to_vector(vector)

        results = self.client.search(
            collection_name=name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=["context_text", "answer_text", "id"],
            with_vectors=False,
        )
        return results