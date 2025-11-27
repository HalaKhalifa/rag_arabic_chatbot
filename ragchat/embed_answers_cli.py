import typer
from datasets import load_from_disk
from tqdm import tqdm
from .config import RAGSettings
from .embeddings import TextEmbedder
from .qdrant_index import QdrantIndex
from .utils import normalize_arabic_text

app = typer.Typer(help="Embed ARCD answers for retrieval evaluation.")

@app.command()
def embed_answers(
    ds_path: str = RAGSettings.clean_arcd_dir,
    collection: str = RAGSettings.answers_col,
    model_name: str = RAGSettings.emb_model,
    force: bool = typer.Option(False, "--force", "-f", help="Recreate answer collection"),
    batch_size: int = typer.Option(32, help="Batch size for embedding"),
):
    """
    Embed ARCD answers into a separate Qdrant collection.
    Not part of our pipeline, used for evaluating embedding model retrieval accuracy.
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
    idx = QdrantIndex(url=RAGSettings.qdrant_url, api_key=RAGSettings.qdrant_api_key)
    test_vec = embedder.embed_text("اختبار")
    dim = len(test_vec)

    if force:
        idx.recreate(collection, dim)
    else:
        idx.ensure_collection(collection, dim)

    print("📝 Extracting answers...")
    answer_texts = []
    payloads = []

    for i, ex in enumerate(split):
        answers = ex.get("answers", {}).get("text", [])
        if answers:
            ans = normalize_arabic_text(answers[0])
            answer_texts.append(ans)
            payloads.append({
                "id": i,
                "answer_text": ans,
                "context": ex.get("context"),
                "question": ex.get("question"),
            })

    print(f"📚 Total answers to embed: {len(answer_texts)}")
    print("⚙️ Embedding answers and uploading...")

    for start in tqdm(range(0, len(answer_texts), batch_size)):
        batch = answer_texts[start : start + batch_size]
        vectors = embedder.embed_batch(batch)
        batch_payloads = payloads[start : start + batch_size]

        idx.upsert(
            name=collection,
            vectors=vectors,
            payloads=batch_payloads,
            start_id=start
        )

    print("🎉 Finished embedding answers for evaluation!")

if __name__ == "__main__":
    app()