import re
from typing import List

_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def normalize(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    return " ".join(text.casefold().split())


def tokenize(text: str) -> List[str]:
    return _TOKEN.findall(normalize(text))
