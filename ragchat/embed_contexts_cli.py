import typer
from datasets import load_from_disk
from .config import settings
from .embeddings import TextEmbedder
from .qdrant_index import QdrantIndex

def main(
    ds_path: str = "data/processed/arcd_clean_prepared",
    collection: str = None,
    model_name: str = None,
):
    """
    Embed ARCD contexts and upsert to Qdrant.
    """
    collection = collection or settings.contexts_col
    model_name = model_name or settings.emb_model

    ds = load_from_disk(ds_path)
    split = ds.get("train") if hasattr(ds, "get") else ds
    contexts = list(split["context"])

    # embed
    emb = TextEmbedder(model_name)
    vecs = emb.encode_passages(contexts)

    # index
    idx = QdrantIndex(settings.qdrant_url, settings.qdrant_api_key)
    idx.recreate(collection, vecs.shape[1])
    payloads = [{"context_text": t, "id": i} for i, t in enumerate(contexts)]
    idx.upsert(collection, vecs, payloads)
    print(f"✅ Upserted {len(contexts)} contexts → {collection}")

if __name__ == "__main__":
    typer.run(main)
