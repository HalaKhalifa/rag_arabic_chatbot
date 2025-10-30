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
    Embed ARCD gold answers (answer[0]) and upsert to Qdrant.
    """
    collection = collection or settings.answers_col
    model_name = model_name or settings.emb_model

    ds = load_from_disk(ds_path)
    split = ds.get("train") if hasattr(ds, "get") else ds

    answers = [
        (ex["answers"]["text"][0] if ex.get("answers", {}).get("text") else "")
        for ex in split
    ]

    emb = TextEmbedder(model_name)
    vecs = emb.encode_passages(answers)

    idx = QdrantIndex(settings.qdrant_url, settings.qdrant_api_key)
    idx.recreate(collection, vecs.shape[1])
    payloads = [{"answer_text": t, "id": i} for i, t in enumerate(answers)]
    idx.upsert(collection, vecs, payloads)
    print(f"✅ Upserted {len(answers)} answers → {collection}")

if __name__ == "__main__":
    typer.run(main)
