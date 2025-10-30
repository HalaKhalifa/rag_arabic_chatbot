import re
from collections import Counter
import sacrebleu


def _tokenize(s: str):
    return re.findall(r"\w+", (s or "").lower())


def f1(pred: str, gold: str) -> float:
    """Token-level F1 between prediction and gold answer."""
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


def bleu(preds, refs):
    """Corpus BLEU using sacrebleu (expects lists)."""
    return sacrebleu.corpus_bleu(preds, [refs]).score
