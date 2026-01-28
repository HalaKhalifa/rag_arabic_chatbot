import re
from typing import List
from ragchat.logger import logger
import hashlib

def clean_unicode(text: str) -> str:
    try:
        if text is None:
            return ""
        text = re.sub(r"[\ud800-\udfff]", "", text)
        text = re.sub(r"[\x00-\x1F\x7F]", " ", text)
        return text
    except Exception as e:
        logger.error(f"clean_unicode() failed: {e}")
        return ""

def normalize_arabic_text(text: str) -> str:
    """
    Basic normalization to unify Arabic forms and remove noisy characters.
    """
    try:
        if not text:
            return ""
    
        text = clean_unicode(text)
        tashkeel = r"[\u0610-\u061A\u064B-\u065F\u06D6-\u06ED]"
        text = re.sub(tashkeel, "", text) # remove tashkeel
        text = re.sub(r"ـ+", "", text) # remove tatweel
        text = re.sub(r"\s+", " ", text).strip() # remove extra whitespace
        return text
    except Exception as e:
        logger.error(f"normalize_arabic_text() failed: {e}")
        return ""

def split_into_sentences(text: str) -> List[str]:
    """
    Simple Arabic sentence splitting based on punctuation.
    """
    try:
        if not text:
            return []

        sentences = re.split(r"[\.!\؟!]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences
    except Exception as e:
        logger.error(f"split_into_sentences() failed: {e}")
        return []


def chunk_sentences(sentences: List[str], group_size: int = 5) -> List[str]:
    """
    Groups sentences into chunks of size N.
    """
    try:
        if not sentences:
            return []

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
    except Exception as e:
        logger.error(f"chunk_sentences() failed: {e}")
        return []

def make_hash_id(text: str) -> str:
    """Return a stable SHA-256 hash for text."""
    try:
        raw_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        # Convert to UUID format: 8-4-4-4-12 since qdrant dont accept 64-char hex string
        uid = f"{raw_hash[0:8]}-{raw_hash[8:12]}-{raw_hash[12:16]}-{raw_hash[16:20]}-{raw_hash[20:32]}"

        return uid
    except Exception as e:
        logger.error(f"make_hash_id() failed: {e}")
        return None
