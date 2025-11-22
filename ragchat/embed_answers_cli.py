import typer
from datasets import load_from_disk
from tqdm import tqdm
from .config import settings
from .embeddings import TextEmbedder
from .qdrant_index import QdrantIndex
from .utils import normalize_arabic_text

app = typer.Typer(help="Embed ARCD answers for retrieval evaluation.")

@app.command()
def embed_answers(
    ds_path: str = settings.clean_arcd_dir,
    collection: str = settings.answers_col,
    model_name: str = settings.emb_model,
    force: bool = typer.Option(False, "--force", "-f", help="Recreate answer collection"),
    batch_size: int = typer.Option(32, help="Batch size for embedding"),
):
    """
    Embed ARCD answers into a separate Qdrant collection.
    Used for evaluating embedding model retrieval accuracy.
    """

    print(f"📥 Loading cleaned dataset from: {ds_path}")
    ds = load_from_disk(ds_path)

    if hasattr(ds, "get"):
        split = (
            ds.get("train")
            or ds.get("validation")
            or next(iter(ds.values()))
        )
    else:
        split = ds

    if "answers" not in split.features:
        raise ValueError("❌ Dataset missing 'answers'. Ensure preprocessing was correct.")

    embedder = TextEmbedder(model_name=model_name)
    idx = QdrantIndex()  # Auto-loads url + api key from settings

    print("📝 Extracting answers...")
    answer_texts = []
    payloads = []

    for i, ex in enumerate(split):
        answers = ex.get("answers", {}).get("text", [])
        if answers:
            raw = answers[0] or ""
            ans = normalize_arabic_text(raw)

            if ans.strip():  # Prevent empty text → zero-dimension vectors
                answer_texts.append(ans)
                payloads.append({
                    "id": i,
                    "answer_text": ans,
                    "context": ex.get("context"),
                    "question": ex.get("question"),
                })

    print(f"📚 Total answers to embed: {len(answer_texts)}")

    if not answer_texts:
        print("⚠️ No answers found to embed. Aborting.")
        return

    example_vecs = embedder.embed_batch([answer_texts[0]])
    if not example_vecs or len(example_vecs[0]) == 0:
        raise RuntimeError("❌ Embedding model returned empty vector for sample answer.")

    dim = len(example_vecs[0])
    print(f"🔢 Detected embedding dimension: {dim}")

    if force:
        idx.recreate(collection, dim)
    else:
        idx.ensure_collection(collection, dim)

    print("🚀 Embedding answers and uploading to Qdrant...")

    for start in tqdm(range(0, len(answer_texts), batch_size)):
        batch = answer_texts[start : start + batch_size]
        vectors = embedder.embed_batch(batch)
        batch_payloads = payloads[start : start + batch_size]

        if not vectors:
            continue

        if len(vectors) != len(batch_payloads):
            raise RuntimeError(
                f"❌ Mismatch: {len(vectors)} vectors vs {len(batch_payloads)} payloads."
            )

        if any(len(v) == 0 for v in vectors):
            raise RuntimeError("❌ Encountered 0-dimension vector in this batch.")

        idx.upsert(
            name=collection,
            vectors=vectors,
            payloads=batch_payloads,
            start_id=start
        )

    print("🎉 Finished embedding answers!")

if __name__ == "__main__":
    app()