import hashlib
from ragchat.config import RAGSettings
from ragchat.core.embeddings import TextEmbedder
from ragchat.storage.qdrant_index import QdrantIndex
from ragchat.data.utils import normalize_arabic_text, make_hash_id


def ingest_text_to_qdrant(text: str):
    """
    Clean, embed, hash, and upsert new text into Qdrant.
    Uses SHA-256 as the stable unique ID for ingestion.
    """

    if not text or not text.strip():
        return {"status": "error", "message": "Empty text."}

    clean = normalize_arabic_text(text)
    uid = make_hash_id(clean)

    embedder = TextEmbedder(RAGSettings.emb_model)
    vector = embedder.embed_text(clean)

    payload = {
        "id": uid,
        "context_text": clean,
        "raw_context": clean,
        "source": "user_ingest",
        "hash": uid,
    }

    index = QdrantIndex(RAGSettings.qdrant_url, RAGSettings.qdrant_api_key)

    # Insert (or overwrite) hashed ID
    index.upsert(
        name=RAGSettings.contexts_col,
        vectors=[vector],
        payloads=[payload],
        start_id=None
    )

    return {
        "status": "ok",
        "inserted_id": uid,
        "text": clean
    }
