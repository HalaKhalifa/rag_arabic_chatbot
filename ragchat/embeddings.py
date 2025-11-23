import torch
from sentence_transformers import SentenceTransformer
from typing import List, Union
from .utils import normalize_arabic_text, clean_unicode
from .config import settings


class TextEmbedder:
    """
    Wrapper around SentenceTransformer model for generating embeddings.
    Ensures:
    - consistent normalization
    - correct pooling
    - correct return shape
    - batching for performance
    """

    def __init__(self, model_name: str = None, device: str = None):
        self.model_name = model_name or settings.emb_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"🔍 Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, device=self.device)

    def embed_text(self, text: str) -> List[float]:
        """
        Embed a single piece of text (normalized).
        Returns a 1D vector list.
        """
        clean = normalize_arabic_text(clean_unicode(text))

        emb = self.model.encode(
            clean,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return emb.tolist()

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Embed a list of texts in batches.
        Returns a list of vectors.
        """
        cleaned = [normalize_arabic_text(clean_unicode(t)) for t in texts]

        embeddings = self.model.encode(
            cleaned,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return [e.tolist() for e in embeddings]