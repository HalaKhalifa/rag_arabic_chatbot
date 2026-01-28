from dataclasses import dataclass
import os

# try importing Django settings (backend mode)
try:
    from django.conf import settings
    DJANGO_LOADED = True
except Exception:
    DJANGO_LOADED = False

def get_setting(name: str, default=None):
    """
    Safely get Django setting if configured,
    otherwise fall back to environment variable.
    """
    if DJANGO_LOADED:
        try:
            return getattr(settings, name)
        except Exception:
            pass
    return os.getenv(name, default)
@dataclass
class RAGSettings:
    raw_arcd_dir: str = "data/raw/arcd_raw"
    clean_arcd_dir: str = "data/processed/arcd_clean"
    emb_model: str = get_setting("EMB_MODEL")
    gen_model: str = get_setting("GEN_MODEL")
    gen_max_new_tokens: int = int(get_setting("GEN_MAX_NEW_TOKENS", 512))
    temperature: float = float(get_setting("GEN_TEMPERATURE", 0.4))
    top_p: float = float(os.getenv("GEN_TOP_P", 0.9))
    qdrant_url: str = get_setting("QDRANT_URL")
    qdrant_api_key: str = get_setting("QDRANT_API_KEY")
    contexts_col: str = os.getenv("QDRANT_CTX_COLLECTION", "arcd_contexts")
    answers_col: str = os.getenv("QDRANT_ANS_COLLECTION", "arcd_answers")
    top_k: int = int(get_setting("TOP_K", 5))
    gemini_api_key: str = get_setting("GEMINI_API_KEY")
