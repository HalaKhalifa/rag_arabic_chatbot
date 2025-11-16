from dataclasses import dataclass
import os
@dataclass
class settings:
    raw_arcd_dir: str = "data/raw/arcd_raw"
    clean_arcd_dir: str = "data/processed/arcd_clean"
    emb_model: str = os.getenv("EMB_MODEL","abdulrahman-nuzha/intfloat-multilingual-e5-large-arabic-fp16")
    gen_model: str = os.getenv("GEN_MODEL", "models/gemini-2.5-flash")
    gen_max_new_tokens: int = int(os.getenv("GEN_MAX_NEW_TOKENS", 256))
    temperature: float = float(os.getenv("GEN_TEMPERATURE", 0.4))
    top_p: float = float(os.getenv("GEN_TOP_P", 0.9))
    qdrant_url: str = os.getenv("QDRANT_URL", "http://qdrant:6333")
    qdrant_api_key: str | None = os.getenv("QDRANT_API_KEY")
    contexts_col: str = os.getenv("QDRANT_CTX_COLLECTION", "arcd_contexts")
    answers_col: str = os.getenv("QDRANT_ANS_COLLECTION", "arcd_answers")
    top_k: int = int(os.getenv("RETR_TOP_K", 5))
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")