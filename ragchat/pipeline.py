from dataclasses import dataclass
from typing import Dict
from .config import settings
from .preprocessing import normalize_arabic_text
from .embeddings import TextEmbedder
from .qdrant_index import QdrantIndex
from .retriever import Retriever
from .generator import Generator


@dataclass
class Services:
    embedder: TextEmbedder
    index: QdrantIndex
    retriever: Retriever
    generator: Generator


class RagPipeline:
    def __init__(self, services: Services):
        self.s = services

    def ask(self, question: str, k: int | None = None) -> Dict:
        """
        Retrieve relevant contexts from Qdrant,
        and generate an Arabic answer using AraT5.
        """
        # Normalize and retrieve
        q = normalize_arabic_text(question)
        hits = self.s.retriever.similar_contexts(q)

        # Filter and select top-K contexts with good scores
        k = k or settings.top_k
        filtered_hits = [h for h in hits if h.get('score', 0) > 0.3]  # Filter low similarity
        contexts = [h["text"] for h in filtered_hits[:k]]
        
        if not contexts and hits:
            contexts = [hits[0]["text"]]
        # Generate answer
        answer = self.s.generator.generate(q, contexts)

        return {
            "question": q,
            "contexts": hits[:k],
            "answer": answer,
        }