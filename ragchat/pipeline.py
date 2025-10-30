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
    def _build_prompt(question: str, contexts: list[str]) -> str:
        """
        Build an Arabic instruction-style prompt for factual Q/A.
        The instruction helps GPT-2 stay concise and answer only from context.
        """
        ctx_block = "\n".join([f"• {c.strip()}" for c in contexts if c.strip()])
        return (
            "أنت مساعد ذكي للإجابة على الأسئلة باللغة العربية.\n"
            "اقرأ النصوص التالية واستخرج منها إجابة قصيرة وواضحة للسؤال.\n"
            "إذا لم تجد الإجابة في النصوص، قل فقط: لا أعلم.\n\n"
            f"النصوص:\n{ctx_block}\n\n"
            f"السؤال: {question.strip()}\n"
            "الإجابة:"
        )

    def ask(self, question: str, k: int | None = None) -> Dict:
        q = normalize_arabic_text(question)
        hits = self.s.retriever.similar_contexts(q)
        contexts = [h["text"] for h in hits[: (k or settings.top_k)]]
        prompt = self._build_prompt(q, contexts)
        answer = self.s.generator.generate(prompt)
        return {"question": q, "contexts": hits, "prompt": prompt, "answer": answer}
