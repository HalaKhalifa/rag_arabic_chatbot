from __future__ import annotations
import numpy as np

# Primary path: Sentence-Transformers
try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception:
    _HAS_ST = False

# Fallback path: plain HF model + mean pooling
from transformers import AutoTokenizer, AutoModel
import torch

def _mean_pooling(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts

class TextEmbedder:
    """
    Robust embedder:
    - Tries Sentence-Transformers.
    - If model isn't ST-compatible, falls back to HF AutoModel + mean pooling.
    Adds E5-style 'query:/passage:' prefixes when encoding (harmless for other models).
    """
    def __init__(self, model_name: str, device: str | None = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._st = None
        if _HAS_ST:
            try:
                self._st = SentenceTransformer(model_name, device=self.device)
            except Exception:
                self._st = None  # will fall back

        if self._st is None:
            self.tok = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)

    @staticmethod
    def _add_prefix(texts, prefix: str):
        return [f"{prefix}: {t}" for t in texts]

    def _encode_hf(self, texts: list[str]) -> np.ndarray:
        batch = self.tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**batch)
            pooled = _mean_pooling(out.last_hidden_state, batch["attention_mask"])
            # normalize
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.cpu().numpy()

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        texts = self._add_prefix(queries, "query")
        if self._st is not None:
            return self._st.encode(texts, normalize_embeddings=True)
        return self._encode_hf(texts)

    def encode_passages(self, passages: list[str]) -> np.ndarray:
        texts = self._add_prefix(passages, "passage")
        if self._st is not None:
            return self._st.encode(texts, normalize_embeddings=True)
        return self._encode_hf(texts)