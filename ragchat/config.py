from dataclasses import dataclass
import os

# try importing Django settings (backend mode)
try:
    from django.conf import settings
    DJANGO_LOADED = True
except Exception:
    DJANGO_LOADED = False

@dataclass
class RAGSettings:
    raw_arcd_dir: str = "data/raw/arcd_raw"
    clean_arcd_dir: str = "data/processed/arcd_clean"

    emb_model: str = (settings.EMB_MODEL if DJANGO_LOADED else os.getenv("EMB_MODEL"))
    gen_model: str = (settings.GEN_MODEL if DJANGO_LOADED else os.getenv("GEN_MODEL"))
    gen_max_new_tokens: int = (settings.GEN_MAX_NEW_TOKENS if DJANGO_LOADED else int(os.getenv("GEN_MAX_NEW_TOKENS", 512)))
    temperature: float = (settings.GEN_TEMPERATURE if DJANGO_LOADED else float(os.getenv("GEN_TEMPERATURE", 0.4)))
    top_p: float = float(os.getenv("GEN_TOP_P", 0.9))
    qdrant_url: str = (settings.QDRANT_URL if DJANGO_LOADED else os.getenv("QDRANT_URL"))
    qdrant_api_key: str = (getattr(settings, "QDRANT_API_KEY", None) if DJANGO_LOADED else os.getenv("QDRANT_API_KEY", None))
    contexts_col: str = os.getenv("QDRANT_CTX_COLLECTION", "arcd_contexts")
    answers_col: str = os.getenv("QDRANT_ANS_COLLECTION", "arcd_answers")
    top_k: int = (settings.TOP_K if DJANGO_LOADED else int(os.getenv("TOP_K", 5)))
    gemini_api_key: str = (settings.GEMINI_API_KEY if DJANGO_LOADED else os.getenv("GEMINI_API_KEY"))