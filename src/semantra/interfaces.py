from typing import Protocol, Sequence

import numpy as np


class EmbeddingModel(Protocol):
    dimension: int

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


class VectorSearcher(Protocol):
    def similarities(self, query: np.ndarray, documents: np.ndarray) -> np.ndarray: ...


class LexicalSearcher(Protocol):
    def fit(self, documents: Sequence[str]) -> None: ...
    def scores(self, query: str) -> Sequence[float]: ...


class ScoreFusion(Protocol):
    def combine(self, semantic: float, lexical: float) -> float: ...
