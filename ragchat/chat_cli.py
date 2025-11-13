import os
from .retriever import Retriever
from .generator import Generator
from .pipeline import ArabicRAGPipeline
from .embeddings import TextEmbedder
from .qdrant_index import QdrantIndex


def main():
    print("💬 Arabic RAG Chatbot (Gemini Edition). Type your question, or /exit to quit.\n")

    # Initialize components
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection_name = os.getenv("QDRANT_COLLECTION", "arcd_answers")
    embedder = TextEmbedder(model_name=os.getenv("EMB_MODEL"))
    index = QdrantIndex(url=qdrant_url)
    retriever = Retriever(embedder=embedder, index=index, collection=collection_name)

    generator = Generator(
        model_name=os.getenv("GEN_MODEL", "models/gemini-2.5-flash"),
        max_new_tokens=int(os.getenv("GEN_MAX_NEW_TOKENS", 512)),
        temperature=float(os.getenv("GEN_TEMPERATURE", 0.4)),
        top_p=float(os.getenv("GEN_TOP_P", 0.9)) if os.getenv("GEN_TOP_P") else 0.9,
    )

    pipeline = ArabicRAGPipeline(
        retriever=retriever,
        generator=generator,
        top_k=int(os.getenv("RETR_TOP_K", 5)),
    )

    # Interactive loop
    while True:
        question = input("سؤالك: ").strip()
        if not question or question.lower() in ["/exit", "exit", "خروج"]:
            print("👋 وداعًا!")
            break

        print("\n🤔 جارٍ البحث عن الإجابة...\n")
        result = pipeline.run(question)

        # Display results
        print("✅ الإجابة:\n", result["answer"], "\n")
        print("--- السياقات الأعلى ---")
        for i, ctx in enumerate(result["contexts"][:3], start=1):
            print(f"• {ctx[:250]}{'...' if len(ctx) > 250 else ''}")
        print(f"\n⏱ الوقت المستغرق: {result['elapsed']:.2f} ثانية\n")
        print("=" * 60)

if __name__ == "__main__":
    main()
