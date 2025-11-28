from typing import List, Sequence
from qdrant_client import QdrantClient
from qdrant_client.http import models
from .config import RAGSettings
from .logger import logger

class QdrantIndex:
    def __init__(self, url: str = None, api_key: str | None = None, timeout: float = 20.0):
        """
        Simple wrapper around QdrantClient for collection management,
        upsert, and search.
        """
        try:
            url = url or RAGSettings.qdrant_url
            api_key = api_key or RAGSettings.qdrant_api_key
            self.client = QdrantClient(url=url, api_key=api_key, prefer_grpc=False, timeout=timeout, check_compatibility=False)
            logger.info(f"Connected to Qdrant at: {url}")
        except Exception as e:
            logger.error(f"Failed to initialize QdrantClient: {e}")
            raise

    def ensure_collection(self, name: str, dim: int):
        """
        Create the collection only if it doesn't already exist.
        """
        try:
            existing = [c.name for c in self.client.get_collections().collections]
        except Exception as e:
            logger.error(f"Failed to get existing Qdrant collections: {e}")
            raise
        try:
            if name not in existing:
                logger.info(f"Creating new Qdrant collection: {name}")
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE)
                )
            else:
                logger.info(f"Qdrant collection already exists: {name}")
        except Exception as e:
            logger.error(f"Failed to create or verify collection '{name}': {e}")
            raise

    def recreate(self, name: str, dim: int):
        try:
            logger.info(f"Recreating collection: {name}")
            self.client.recreate_collection(
                collection_name=name,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE)
            )
        except Exception as e:
            logger.error(f"Failed to recreate collection '{name}': {e}")
            raise

    def _to_vector(self, v: Sequence[float]) -> List[float]:
        """
        Safely convert numpy arrays or lists/tuples to a plain Python list.
        """
        try:
            return v.tolist() if hasattr(v, "tolist") else list(v)
        except Exception as e:
            logger.error(f"Failed to convert vector to list: {e}")
            raise

    def upsert(self, name: str, vectors, payloads, start_id: int):
        """
        Upsert a batch of points into Qdrant with globally unique IDs.
        - vectors: iterable of embedding vectors (list[list[float]] or np.ndarray)
        - payloads: same length as vectors, each is a dict
        """
        try:
            points = []
            for offset, (v, payload) in enumerate(zip(vectors, payloads)):
                points.append(
                    models.PointStruct(
                        id=start_id + offset,
                        vector=self._to_vector(v),
                        payload=payload,
                    )
                )

            self.client.upsert(collection_name=name, points=points, wait=True)

            logger.info(f"Upserted {len(points)} points into '{name}' (IDs {start_id}–{start_id + len(points) - 1})")

        except Exception as e:
            logger.error(f"Failed to upsert points to collection '{name}': {e}")
            raise

    def search(self, name: str, vector, top_k: int = 5):
        """
        Search top_k nearest neighbors for a given query vector.
        """
        try:
            query_vector = self._to_vector(vector)
            query = models.NearestQuery(nearest=query_vector)

            results = self.client.query_points(
                collection_name=name,
                query=query,
                limit=top_k,
                with_vectors=False,
                with_payload=True,
            )

            return results.points
        except Exception as e:
            logger.error(f"Qdrant search failed for collection '{name}': {e}")
            return []   # safer fallback
