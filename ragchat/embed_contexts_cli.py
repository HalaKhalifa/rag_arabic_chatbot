import typer
from datasets import load_from_disk
from tqdm import tqdm
from .config import settings
from .embeddings import TextEmbedder
from .qdrant_index import QdrantIndex
from .utils import normalize_arabic_text

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

    idx = QdrantIndex()

    print("🔎 Detecting embedding dimension...")
    sample_vec = embedder.embed_batch(["مثال"])
    if not sample_vec or len(sample_vec[0]) == 0:
        raise RuntimeError("❌ Embedding model returned empty vector.")

    dim = len(sample_vec[0])
    print(f"🔢 Detected embedding dimension: {dim}")

    if force:
        idx.recreate(collection, dim)
    else:
        idx.ensure_collection(collection, dim)

    all_texts = []
    all_payloads = []
    print("🧩 Flattening chunks...")

    for i, ex in enumerate(split):
        for j, chunk in enumerate(ex["chunks"]):
            text = normalize_arabic_text(chunk or "")

            if not text.strip():
                continue

            all_texts.append(text)
            all_payloads.append({
                "id": i,
                "chunk_index": j,
                "context_text": text,
                "answer_text": ex["answers"]["text"][0] if ex.get("answers") else None,
                "raw_context": ex["context"],
                "question": ex["question"],
            })

    print(f"📚 Total chunks to embed: {len(all_texts)}")
    print("🚀 Embedding and uploading to Qdrant...")

    for start in tqdm(range(0, len(all_texts), batch_size)):
        batch = all_texts[start:start + batch_size]
        batch_payloads = all_payloads[start:start + batch_size]

        vectors = embedder.embed_batch(batch)

        if not vectors:
            continue
        if len(vectors) != len(batch_payloads):
            raise RuntimeError("❌ Vector/payload count mismatch.")
        if any(len(v) == 0 for v in vectors):
            raise RuntimeError("❌ Zero-dimension vector detected.")

        idx.upsert(
            name=collection,
            vectors=vectors,
            payloads=batch_payloads,
            start_id=start
        )

    print("🎉 All chunks successfully embedded and stored in Qdrant!")

if __name__ == "__main__":
    app()