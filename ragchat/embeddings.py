import torch
from sentence_transformers import SentenceTransformer
from typing import List
from .utils import normalize_arabic_text, clean_unicode
from .config import settings

class TextEmbedder:
    """
    Clean + fast embedding wrapper.
    - Uses SentenceTransformer batching
    - Applies unicode + Arabic normalization
    - Works with large Qdrant batches
    """

    def __init__(self, model_name: str = None, device: str = None):
        self.model_name = model_name or settings.emb_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"🔍 Loading embedding model: {self.model_name} on device={self.device}")
        self.model = SentenceTransformer(self.model_name, device=self.device)

    def _clean(self, text: str) -> str:
        """
        Apply both unicode cleaning + Arabic normalization.
        """
        if not isinstance(text, str):
            text = str(text)
        text = clean_unicode(text)
        text = normalize_arabic_text(text)
        return text

    def embed_text(self, text: str) -> List[float]:
        """
        Embed a single piece of text.
        Returns a 1D vector list.
        """
        clean = self._clean(text)

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
        cleaned = [self._clean(t) for t in texts]

        embeddings = self.model.encode(
            cleaned,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return [e.tolist() for e in embeddings]