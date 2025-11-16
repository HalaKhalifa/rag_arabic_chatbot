from typing import List, Dict, Any
from .embeddings import TextEmbedder
from .qdrant_index import QdrantIndex
from .utils import normalize_arabic_text


class Retriever:
    """
    Retrieves top similar context chunks from Qdrant based on the user's question embedding.
    """
    def __init__(self, embedder: TextEmbedder, index: QdrantIndex,
                 collection: str, top_k: int = 5):
        self.embedder = embedder
        self.index = index
        self.collection = collection
        self.top_k = top_k

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Given a user query:
        - normalize it
        - embed it
        - search Qdrant
        - return ranked contexts
        """
        clean_query = normalize_arabic_text(query)
        vector = self.embedder.embed_text(clean_query)
        results = self.index.search(
            name=self.collection,
            vector=vector,
            top_k=self.top_k
        )
        formatted = []
        for hit in results:
            payload = hit.payload or {}

            formatted.append({
                "score": hit.score,
                "chunk": payload.get("context_text"),
                "chunk_index": payload.get("chunk_index"),
                "raw_context": payload.get("raw_context"),
                "question": payload.get("question"),
                "answer": payload.get("answer_text"),
            })
        return formatted