import typer
from datasets import load_from_disk
from tqdm import tqdm
from ragchat.config import RAGSettings
from ragchat.core.embeddings import TextEmbedder
from ragchat.storage.qdrant_index import QdrantIndex
from ragchat.logger import logger

app = typer.Typer(help="Embed ARCD contexts and store them in Qdrant.")

def load_dataset_split(ds_path: str):
    """Load ARCD dataset and return a single split."""
    try:
        logger.info(f"Loading cleaned dataset from {ds_path}")
        ds = load_from_disk(ds_path)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

    # dataset may contain train/validation, order matters.
    if hasattr(ds, "get"):
        if ds.get("train") is not None:
            return ds["train"]
        elif ds.get("validation") is not None:
            return ds["validation"]
        else:
            return next(iter(ds.values()))
    else:
        return ds

def prepare_qdrant(embedder, idx, collection: str, force: bool):
    """Ensure Qdrant collection exists with correct dimensions."""
    try:
        example_vec = embedder.embed_text("مثال")
        dim = len(example_vec)
    except Exception as e:
        logger.error(f"Failed to compute embedding dimension: {e}")
        raise

    try:
        if force:
            logger.info(f"Recreating Qdrant collection '{collection}'")
            idx.recreate(collection, dim)
        else:
            logger.info(f"Ensuring collection '{collection}' exists")
            idx.ensure_collection(collection, dim)
    except Exception as e:
        logger.error(f"Failed to prepare Qdrant collection: {e}")
        raise

    return dim


def extract_chunks(split):
    """Flatten chunk lists into text + payload lists."""
    all_texts = []
    all_payloads = []

    for i, ex in enumerate(split):

        # handle missing or empty answers
        answer_list = ex.get("answers", {}).get("text", [])
        answer_text = answer_list[0] if answer_list else None

        for j, chunk in enumerate(ex["chunks"]):
            all_texts.append(chunk)
            all_payloads.append({
                "id": i,
                "chunk_index": j,
                "context_text": chunk,
                "answer_text": answer_text,
                "raw_context": ex.get("context"),
                "question": ex.get("question"),
            })

    return all_texts, all_payloads

@app.command()
def embed_contexts(
    ds_path: str = RAGSettings.clean_arcd_dir,
    collection: str = RAGSettings.contexts_col,
    model_name: str = RAGSettings.emb_model,
    force: bool = typer.Option(False, "--force", "-f", help="Recreate Qdrant collection"),
    batch_size: int = typer.Option(32, help="Embedding batch size"),
):
    """Embed all context chunks and upsert into Qdrant."""
    try:
        split = load_dataset_split(ds_path)

        if "chunks" not in split.features:
            raise ValueError("Dataset missing 'chunks'. Run preprocessing first.")

        embedder = TextEmbedder(model_name)
        idx = QdrantIndex(url=RAGSettings.qdrant_url,api_key=RAGSettings.qdrant_api_key)

        prepare_qdrant(embedder, idx, collection, force)

        logger.info("Flattening chunks...")
        all_texts, all_payloads = extract_chunks(split)

        logger.info(f"Total chunks to embed: {len(all_texts)}")
        logger.info("Embedding and uploading chunks...")

        # batch embedding
        for start in tqdm(range(0, len(all_texts), batch_size)):
            batch = all_texts[start:start + batch_size]
            vectors = embedder.embed_batch(batch)
            batch_payloads = all_payloads[start:start + batch_size]

            idx.upsert(
                name=collection,
                vectors=vectors,
                payloads=batch_payloads,
                start_id=start
            )

        logger.info("All chunks embedded and stored successfully!")

    except Exception as e:
        logger.error(f"Context embedding failed: {e}")


if __name__ == "__main__":
    app()
