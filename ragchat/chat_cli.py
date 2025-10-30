import typer
from .config import settings
from .embeddings import TextEmbedder
from .qdrant_index import QdrantIndex
from .retriever import Retriever
from .generator import Generator
from .pipeline import RagPipeline, Services

def main(
    top_k: int = typer.Option(5, "--k", "-k", help="Top-K contexts to use."),
    max_new_tokens: int = typer.Option(64, "--max-new-tokens"),
    temperature: float = typer.Option(0.7, "--temperature"),
    top_p: float = typer.Option(0.95, "--top-p"),
):
    emb = TextEmbedder(settings.emb_model)
    idx = QdrantIndex(settings.qdrant_url, settings.qdrant_api_key)
    retr = Retriever(emb, idx, settings.contexts_col, top_k)
    gen = Generator(settings.gen_model, max_new_tokens, temperature, top_p)
    pipe = RagPipeline(Services(emb, idx, retr, gen))

    print("\n💬 Arabic RAG Chatbot. Type your question, or /exit to quit.\n")
    while True:
        try:
            q = input("سؤالك: ").strip()
            if q.lower() in {"/exit", "exit", "quit"}:
                break
            out = pipe.ask(q, k=top_k)
            print("\n--- السياق الأعلى ---")
            for h in out["contexts"]:
                print(f"• {h['text'][:120]}… (score={h['score']:.3f})")
            print("\n--- الإجابة ---")
            print(out["answer"], "\n")
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    typer.run(main)
