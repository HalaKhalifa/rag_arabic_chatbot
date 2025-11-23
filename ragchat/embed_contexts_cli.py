import typer
from datasets import load_from_disk
from tqdm import tqdm
from .config import settings
from .embeddings import TextEmbedder
from .qdrant_index import QdrantIndex

app = typer.Typer(help="Embed ARCD contexts and store them in Qdrant.")

@app.command()
def embed_contexts(
    ds_path: str = settings.clean_arcd_dir,
    collection: str = settings.contexts_col,
    model_name: str = settings.emb_model,
    force: bool = typer.Option(False, "--force", "-f", help="Recreate Qdrant collection"),
    batch_size: int = typer.Option(32, help="Embedding batch size"),
):
    """
    Load the cleaned dataset (with 'chunks' field),
    embed all chunks, and upsert into Qdrant.
    """
    print(f"📥 Loading cleaned dataset from: {ds_path}")
    ds = load_from_disk(ds_path)

    if hasattr(ds, "get"):
        if ds.get("train") is not None:
            split = ds["train"]
        elif ds.get("validation") is not None:
            split = ds["validation"]
        else:
            split = next(iter(ds.values()))
    else:
        split = ds

    if "chunks" not in split.features:
        raise ValueError("❌ Dataset does not contain 'chunks'. Run preprocessing first.")

    embedder = TextEmbedder(model_name=model_name)
    idx = QdrantIndex(url=settings.qdrant_url, api_key=settings.qdrant_api_key)

    example_vec = embedder.embed_text("مثال")
    dim = len(example_vec)

    if force:
        idx.recreate(collection, dim)
    else:
        idx.ensure_collection(collection, dim)

    all_texts = []
    all_payloads = []

    print("🧩 Flattening chunks..")

    for i, ex in enumerate(split):
        for j, chunk in enumerate(ex["chunks"]):
            all_texts.append(chunk)
            all_payloads.append({
                "id": i,
                "chunk_index": j,
                "context_text": chunk,
                "answer_text": ex["answers"]["text"][0] if ex.get("answers") else None,
                "raw_context": ex["context"],
                "question": ex["question"],
            })

    print(f"📚 Total chunks to embed: {len(all_texts)}")
    print("⚙️ Embedding all chunks and uploading to Qdrant...")

    for start in tqdm(range(0, len(all_texts), batch_size)):
        batch = all_texts[start : start + batch_size]
        vectors = embedder.embed_batch(batch)
        batch_payloads = all_payloads[start : start + batch_size]

        idx.upsert(
            name=collection,
            vectors=vectors,
            payloads=batch_payloads,
            start_id=start
        )

    print("🎉 All chunks embedded and stored in QdrantSuccessfully!")

if __name__ == "__main__":
    app()