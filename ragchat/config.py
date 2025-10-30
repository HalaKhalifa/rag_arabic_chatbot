from dataclasses import dataclass
import os

@dataclass
class Settings:
    processed_ds_dir: str = os.getenv("PROCESSED_DS_DIR", "data/processed/arcd_clean")
    emb_model: str = os.getenv("EMB_MODEL", "abdulrahman-nuzha/intfloat-multilingual-e5-large-arabic-fp16")
    gen_model: str = os.getenv("GEN_MODEL", "aubmindlab/aragpt2-base")
    max_new_tokens: int = int(os.getenv("GEN_MAX_NEW_TOKENS", 64))
    temperature: float = float(os.getenv("GEN_TEMPERATURE", 0.7))
    top_p: float = float(os.getenv("GEN_TOP_P", 0.95))
    qdrant_url: str = os.getenv("QDRANT_URL", "http://qdrant:6333")
    qdrant_api_key: str | None = os.getenv("QDRANT_API_KEY")
    contexts_col: str = os.getenv("QDRANT_CTX_COLLECTION", "arcd_contexts")
    answers_col: str = os.getenv("QDRANT_ANS_COLLECTION", "arcd_answers")
    top_k: int = int(os.getenv("RETR_TOP_K", 5))

settings = Settings()
