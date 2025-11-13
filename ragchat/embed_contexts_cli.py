import typer
from datasets import load_from_disk
from .config import settings
from .embeddings import TextEmbedder
from .qdrant_index import QdrantIndex

def main(
    ds_path: str = "data/processed/arcd_clean_prepared",
    collection: str = None,
    model_name: str = None,
    force: bool = typer.Option(False, "--force", "-f", help="Force recreation of the Qdrant collection."),
):
    """
    Embed ARCD contexts and upsert to Qdrant.
    Use --force to drop and recreate the collection from scratch.
    """
    collection = collection or settings.contexts_col
    model_name = model_name or settings.emb_model

    ds = load_from_disk(ds_path)
    split = ds.get("train") if hasattr(ds, "get") else ds
    contexts = list(split["context"])
    
    emb = TextEmbedder(model_name)
    vecs = emb.encode_passages(contexts)
    idx = QdrantIndex(settings.qdrant_url, settings.qdrant_api_key)

    if force:
        idx.recreate(collection, vecs.shape[1])
    else:
        idx.ensure_collection(collection, vecs.shape[1])

    payloads = [{"context_text": t, "id": i} for i, t in enumerate(contexts)]
    idx.upsert(collection, vecs, payloads)

    print(f"✅ Upserted {len(contexts)} contexts → {collection}")
    if force:
        print("♻️ Collection recreated from scratch.")

if __name__ == "__main__":
    typer.run(main)
