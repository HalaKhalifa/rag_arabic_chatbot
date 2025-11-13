import time
from .retriever import Retriever
from .generator import Generator
from .utils import clean_text


class ArabicRAGPipeline:
    """
    Simple RAG pipeline for Arabic QA:
      1. Retrieve top contexts from Qdrant
      2. Build prompt for Gemini
      3. Generate Arabic answer
    """

    def __init__(self, retriever: Retriever, generator: Generator, top_k: int = 5):
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k

    def run(self, question: str) -> dict:
        start = time.time()
        question_clean = clean_text(question)

        # Retrieve
        results = self.retriever.similar_contexts(question_clean)
        contexts = [r["text"] for r in results if r.get("text")]

        if not contexts:
            return {
                "question": question,
                "answer": "لم أجد سياقًا مناسبًا للإجابة على هذا السؤال.",
                "contexts": [],
                "elapsed": time.time() - start,
            }

        # Generate
        answer = self.generator.generate(question_clean, contexts)

        elapsed = time.time() - start
        return {
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "elapsed": elapsed,
        }
