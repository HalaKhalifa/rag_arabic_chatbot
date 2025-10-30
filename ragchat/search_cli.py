import typer
from .config import settings
from .embeddings import TextEmbedder
from .qdrant_index import QdrantIndex
from .retriever import Retriever

def main(
    q: str = typer.Option(..., "-q", "--q", help="Arabic question to search for."),
    k: int = typer.Option(5, "-k", "--k", help="Number of results to return."),
):
    """Search Qdrant for contexts most similar to a question."""
    emb = TextEmbedder(settings.emb_model)
    idx = QdrantIndex(settings.qdrant_url, settings.qdrant_api_key)
    retr = Retriever(emb, idx, settings.contexts_col, k)
    hits = retr.similar_contexts(q)

    print(f"\n🔍 Top {len(hits)} results for: {q}\n")
    for i, h in enumerate(hits, 1):
        snippet = h["text"][:150].replace("\n", " ")
        print(f"{i:02d}. score={h['score']:.3f}  {snippet}…\n")

if __name__ == "__main__":
    typer.run(main)
