import re

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


def load_and_clean(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    return clean_text(raw)