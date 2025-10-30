from dataclasses import dataclass
from typing import Dict, List
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

    @staticmethod
    def _build_prompt(question: str, contexts: List[str]) -> str:
        ctx_block = "\n\n".join([f"- {c}" for c in contexts])
        return (
            "أنت مساعد للإجابة على الأسئلة باللغة العربية بالاعتماد على السياق التالي.\n"
            "إذا لم تجد الإجابة في السياق، قل أنك غير متأكد.\n\n"
            f"السياق:\n{ctx_block}\n\n"
            f"السؤال: {question}\n"
            "الإجابة: "
        )

    def ask(self, question: str, k: int | None = None) -> Dict:
        q = normalize_arabic_text(question)
        hits = self.s.retriever.similar_contexts(q)
        contexts = [h["text"] for h in hits[: (k or settings.top_k)]]
        prompt = self._build_prompt(q, contexts)
        answer = self.s.generator.generate(prompt)
        return {"question": q, "contexts": hits, "prompt": prompt, "answer": answer}
