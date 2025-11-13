from typing import List, Dict
from .embeddings import TextEmbedder
from .qdrant_index import QdrantIndex

class Retriever:
    """Encodes a question and retrieves top-K similar contexts."""
    def __init__(self, embedder: TextEmbedder, index: QdrantIndex,
                 collection: str, top_k: int = 5):
        self.embedder = embedder
        self.index = index
        self.collection = collection
        self.top_k = top_k
        self._cache = {}

    def similar_contexts(self, question: str) -> List[Dict]:
        if question in self._cache:
            return self._cache[question]
        qv = self.embedder.encode_queries([question])[0]
        hits = self.index.search(self.collection, qv, top_k=self.top_k)
        results = [{
            "score": float(h.score),
            "text": h.payload.get("context_text") or h.payload.get("answer_text") or "",
            "id": h.payload.get("id"),
        } for h in hits]
        self._cache[question] = results
        return results