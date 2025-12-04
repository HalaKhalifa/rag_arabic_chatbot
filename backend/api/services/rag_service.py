import hashlib
from ragchat.config import RAGSettings
from ragchat.core.embeddings import TextEmbedder
from ragchat.storage.qdrant_index import QdrantIndex
from ragchat.data.utils import normalize_arabic_text


def ingest_text_to_qdrant(text: str):
    """
    Clean, embed, hash, and upsert new text into Qdrant.
    Uses SHA-256 as the stable unique ID for ingestion.
    """

    if not text or not text.strip():
        return {"status": "error", "message": "Empty text."}

    clean = normalize_arabic_text(text)
    raw_hash = hashlib.sha256(clean.encode("utf-8")).hexdigest()

    # Convert to UUID format: 8-4-4-4-12 since qdrant dont accept 64-char hex string
    uid = f"{raw_hash[0:8]}-{raw_hash[8:12]}-{raw_hash[12:16]}-{raw_hash[16:20]}-{raw_hash[20:32]}"

    embedder = TextEmbedder(RAGSettings.emb_model)
    vector = embedder.embed_text(clean)

    payload = {
        "context_text": clean,
        "raw_context": clean,
        "source": "user_ingest",
    }

    index = QdrantIndex(RAGSettings.qdrant_url, RAGSettings.qdrant_api_key)

    # Insert (or overwrite) hashed ID
    index.client.upsert(
        collection_name=RAGSettings.contexts_col,
        points=[
            {
                "id": uid,
                "vector": vector,
                "payload": payload
            }
        ],
        wait=True,
    )

    return {
        "status": "ok",
        "inserted_id": uid,
        "text": clean
    }
