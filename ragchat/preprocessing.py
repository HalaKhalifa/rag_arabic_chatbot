import re

# arabic diacritics
_ARABIC_DIAC = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")

def normalize_arabic_text(text: str) -> str:
    if not text:
        return ""
    t = text
    t = _ARABIC_DIAC.sub("", t)        # remove tashkeel
    t = t.replace("\u0640", "")        # remove tatweel
    t = t.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    t = t.replace("ى", "ي").replace("ۀ", "ه").replace("ؤ", "و").replace("ئ", "ي")
    return re.sub(r"\s+", " ", t).strip()
