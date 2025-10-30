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

    def similar_contexts(self, question: str) -> List[Dict]:
        qv = self.embedder.encode_queries([question])[0]
        hits = self.index.search(self.collection, qv, top_k=self.top_k)
        results = []
        for h in hits:
            pl = h.payload or {}
            results.append({
                "score": float(h.score),
                "text": pl.get("context_text") or pl.get("answer_text") or "",
                "id": pl.get("id"),
            })
        return results