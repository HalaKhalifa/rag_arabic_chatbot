import re, unicodedata

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"[\ud800-\udfff]", "", s)   # strip invalid surrogates
    s = re.sub(r"\s+", " ", s).strip()
    return s
