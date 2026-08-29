import re
import unicodedata
from typing import List

# ``\w`` omits combining marks used by scripts such as Devanagari. Include
# Unicode combining marks explicitly so lexical matching does not split words.
_TOKEN = re.compile(
    r"(?:[^\W_]|[\u0300-\u036f\u0900-\u097f])+(?:['’](?:[^\W_]|[\u0300-\u036f\u0900-\u097f])+)?",
    re.UNICODE,
)


def normalize(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    return " ".join(unicodedata.normalize("NFKC", text.casefold()).split())


def tokenize(text: str) -> List[str]:
    return _TOKEN.findall(normalize(text))
