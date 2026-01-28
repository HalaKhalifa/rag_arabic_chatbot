import typer
from datasets import load_from_disk
from tqdm import tqdm
from ragchat.config import RAGSettings
from ragchat.core.embeddings import TextEmbedder
from ragchat.storage.qdrant_index import QdrantIndex
from ragchat.data.utils import normalize_arabic_text, make_hash_id
from ragchat.logger import logger

app = typer.Typer(help="Embed ARCD answers for retrieval evaluation.")

def load_dataset_split(ds_path: str):
    """Load dataset safely and return a single split."""
    try:
        logger.info(f"Loading cleaned dataset from {ds_path}")
        ds = load_from_disk(ds_path)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

    # The dataset may include multiple splits (train/validation/test),
    # so we check for them in preferred order.
    if hasattr(ds, "get"):
        return (
            ds.get("train")
            or ds.get("validation")
            or next(iter(ds.values()))
        )
    else:
        return ds

def prepare_qdrant_collection(embedder, idx, collection: str, force: bool):
    """Ensure Qdrant collection exists with correct embedding dimension."""
    try:
        test_vec = embedder.embed_text("اختبار")
        dim = len(test_vec)
    except Exception as e:
        logger.error(f"Failed to compute embedding dimension: {e}")
        raise

    try:
        if force:
            logger.info(f"Recreating Qdrant collection '{collection}'")
            idx.recreate(collection, dim)
        else:
            logger.info(f"Ensuring Qdrant collection '{collection}' exists")
            idx.ensure_collection(collection, dim)
    except Exception as e:
        logger.error(f"Failed to create/verify Qdrant collection: {e}")
        raise

    return dim


def extract_answers(split):
    """Extract answer texts + payloads from dataset split."""
    answer_texts = []
    payloads = []

    for i, ex in enumerate(split):
        answers = ex.get("answers", {}).get("text", [])
        if answers:
            ans = normalize_arabic_text(answers[0])
            hash_id = make_hash_id(ans)

            answer_texts.append(ans)
            payloads.append({
                "id": hash_id,
                "original_example_id": i,
                "answer_text": ans,
                "context": ex.get("context"),
                "question": ex.get("question"),
                "hash": hash_id,
            })

    return answer_texts, payloads

@app.command()
def embed_answers(
    ds_path: str = RAGSettings.clean_arcd_dir,
    collection: str = RAGSettings.answers_col,
    model_name: str = RAGSettings.emb_model,
    force: bool = typer.Option(False, "--force", "-f", help="Recreate answer collection"),
    batch_size: int = typer.Option(32, help="Batch size for embedding"),
):
    """Embed ARCD answers into a separate Qdrant collection."""
    try:
        # load + select split
        split = load_dataset_split(ds_path)

        if "answers" not in split.features:
            raise ValueError("Dataset missing 'answers'.")

        # initialize the embedder + Qdrant
        embedder = TextEmbedder(model_name=model_name)
        idx = QdrantIndex(url=RAGSettings.qdrant_url, api_key=RAGSettings.qdrant_api_key)

        prepare_qdrant_collection(embedder, idx, collection, force)

        # extract answer texts
        logger.info("Extracting answers...")
        answer_texts, payloads = extract_answers(split)

        logger.info(f"Total answers to embed: {len(answer_texts)}")
        logger.info("Embedding answers and uploading...")

        # batch embedding + upload
        for start in tqdm(range(0, len(answer_texts), batch_size)):
            batch = answer_texts[start: start + batch_size]
            vectors = embedder.embed_batch(batch)
            batch_payloads = payloads[start: start + batch_size]

            idx.upsert(
                name=collection,
                vectors=vectors,
                payloads=batch_payloads,
                start_id=None
            )

        logger.info("Finished embedding answers!")

    except Exception as e:
        logger.error(f"Answer embedding failed: {e}")


if __name__ == "__main__":
    app()
