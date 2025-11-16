import re
from typing import List

def clean_unicode(text: str) -> str:
    text = re.sub(r"[\ud800-\udfff]", "", text)
    text = re.sub(r"[\x00-\x1F\x7F]", " ", text)
    return text

def normalize_arabic_text(text: str) -> str:
    """
    Basic normalization to unify Arabic forms and remove noisy characters.
    """
    if not text:
        return ""
    text = clean_unicode(text)
    tashkeel = r"[\u0610-\u061A\u064B-\u065F\u06D6-\u06ED]"
    text = re.sub(tashkeel, "", text) # remove tashkeel
    text = re.sub(r"ـ+", "", text) # remove tatweel
    text = re.sub(r"\s+", " ", text).strip() # remove extra whitespace
    return text

def split_into_sentences(text: str) -> List[str]:
    """
    Simple Arabic sentence splitting based on punctuation.
    """
    if not text:
        return []

    sentences = re.split(r"[\.!\؟!]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    return sentences

def chunk_sentences(sentences: List[str], group_size: int = 5) -> List[str]:
    """
    Groups sentences into chunks of size N.
    """
    chunks = []
    current = []

    for s in sentences:
        current.append(s)
        if len(current) == group_size:
            chunks.append(" ".join(current))
            current = []

    if current:
        chunks.append(" ".join(current))

    return chunks