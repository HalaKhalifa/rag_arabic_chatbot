from typing import List, Dict, Any, Optional
from .embeddings import TextEmbedder
from .retriever import Retriever
from .generator import Generator
from .config import settings

class RagPipeline:
    """
    Full end-to-end Arabic RAG pipeline:
    - Embed user question
    - Retrieve top-k chunks from Qdrant
    - Generate answer using Gemini with those contexts
    """

    def __init__(
        self,
        embedder: Optional[TextEmbedder] = None,
        retriever: Optional[Retriever] = None,
        generator: Optional[Generator] = None,
        top_k: Optional[int] = None,
    ):
        self.embedder = embedder or TextEmbedder(settings.emb_model)
        self.retriever = retriever
        self.generator = generator or Generator(settings.gen_model)
        self.top_k = top_k or settings.top_k
        if self.retriever is None:
            raise ValueError("Retriever must be provided to RagPipeline.")

    def answer(self, question: str) -> Dict[str, Any]:
        """
        Execute the full RAG flow:
        1. Retrieve top-k contexts
        2. Generate answer from Gemini
        3. Return both answer + contexts
        """

        # Retrieve
        contexts = self.retriever.retrieve(question)
        context_texts = []
        for c in contexts:
            txt = (
                c.get("chunk")
                or c.get("context_text")
                or c.get("raw_context")
                or None
            )
            if txt:
                trimmed = txt[:350]
                context_texts.append(trimmed)

        # Generate
        answer = self.generator.generate(question, contexts=context_texts)

        return {
            "question": question,
            "answer": answer,
            "retrieved_contexts": contexts,
        }