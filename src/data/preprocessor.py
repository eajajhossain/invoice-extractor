import re

def preprocess_text(text: str) -> list[str]:
    """
    Normalize and tokenize OCR text.
    MUST mirror training-time preprocessing.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9.$:/\- ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()
