from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class PredictionCandidate:
    class_name: str
    confidence: float
    semantic_score: float
    lexical_score: float


@dataclass(frozen=True)
class Prediction:
    class_name: Optional[str]
    confidence: float
    semantic_score: float
    lexical_score: float
    margin: float
    top_k: List[PredictionCandidate]
    is_unknown: bool

    @property
    def label(self) -> Optional[str]:
        return self.class_name
