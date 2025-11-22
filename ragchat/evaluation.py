import re
from collections import Counter
from typing import List
import sacrebleu

from .utils import normalize_arabic_text


def normalize_text(s: str) -> str:
    """
    Basic normalization for evaluation.

    - Handles None safely
    - Applies the same Arabic normalization used in the pipeline
    - Lowercases and strips extra punctuation/whitespace
    """
    s = s or ""
    s = normalize_arabic_text(s)
    s = s.lower()
    # Keep word characters + Arabic letters + digits + whitespace
    s = re.sub(r"[^\w\sء-ي0-9]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _tokenize(s: str) -> List[str]:
    # Mix of \w+ and Arabic sequences
    return re.findall(r"\w+|[ء-ي]+", normalize_text(s))


def f1(pred: str, gold: str) -> float:
    """
    Token-level F1 between prediction and gold answer.

    Works on normalized Arabic text to be consistent with the rest of the pipeline.
    """
    p, g = _tokenize(pred), _tokenize(gold)
    if not p or not g:
        return 0.0
    p_c, g_c = Counter(p), Counter(g)
    overlap = sum((p_c & g_c).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def bleu(preds: List[str], refs: List[str]) -> float:
    """
    Corpus BLEU using sacrebleu (expects lists of strings).

    Args:
        preds: list of model predictions
        refs:  list of reference / gold answers (same length as preds)
    """
    if not preds or not refs:
        return 0.0
    # sacrebleu expects refs as list-of-lists
    return sacrebleu.corpus_bleu(preds, [refs]).score