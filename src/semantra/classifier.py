import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .bm25 import BM25
from .config import ClassifierConfig
from .embeddings import OnnxEmbeddingModel
from .fusion import WeightedScoreFusion
from .preprocessing import normalize
from .types import Prediction, PredictionCandidate
from .vectors import NumpyVectorSearcher


class Classifier:
    """Few-shot classifier combining normalized cosine and BM25 scores."""

    def __init__(
        self,
        embedding_model=None,
        config: Optional[ClassifierConfig] = None,
        lexical_searcher=None,
        vector_searcher=None,
        score_fusion=None,
    ):
        self.embedding_model = embedding_model or OnnxEmbeddingModel()
        self.config = config or ClassifierConfig()
        self._classes: Dict[str, List[str]] = {}
        self._examples: List[str] = []
        self._owners: List[str] = []
        self._embeddings = np.empty(
            (0, getattr(self.embedding_model, "dimension", 0)), dtype=np.float32
        )
        self._bm25 = lexical_searcher or BM25([])
        self._vector_searcher = vector_searcher or NumpyVectorSearcher()
        self._fusion = score_fusion or WeightedScoreFusion(
            self.config.semantic_weight, self.config.lexical_weight
        )

    @property
    def classes(self) -> List[str]:
        return list(self._classes)

    def add_class(self, name: str, examples: Sequence[str]) -> "Classifier":
        if not isinstance(name, str) or not name.strip():
            raise ValueError("class name must be non-empty")
        if name in self._classes:
            raise ValueError("class already exists: " + name)
        clean = [normalize(x) for x in examples]
        if not clean:
            raise ValueError("each class needs at least one example")
        if len(clean) > self.config.examples_per_class:
            raise ValueError("too many examples for class")
        self._classes[name] = clean
        self._examples.extend(clean)
        self._owners.extend([name] * len(clean))
        if isinstance(self._bm25, BM25):
            self._bm25 = BM25(self._examples)
        else:
            self._bm25.fit(self._examples)
        self._embeddings = self.embedding_model.embed(self._examples).astype(np.float32)
        return self

    def predict(self, query: str, top_k: Optional[int] = None) -> Prediction:
        if not self._examples:
            raise RuntimeError("add at least one class before predicting")
        text = normalize(query)
        q = self.embedding_model.embed([text])[0].astype(np.float32)
        semantic = self._vector_searcher.similarities(q, self._embeddings)
        lexical = np.asarray(self._bm25.scores(text))
        scores = {}
        parts = {}
        for cls in self._classes:
            indices = [i for i, owner in enumerate(self._owners) if owner == cls]
            ss, ls = (
                float(max(semantic[i] for i in indices)),
                float(max(lexical[i] for i in indices)),
            )
            parts[cls] = (ss, ls)
            scores[cls] = self._fusion.combine(ss, ls)
        ordered = sorted(scores, key=scores.get, reverse=True)
        candidates = [
            PredictionCandidate(c, scores[c], parts[c][0], parts[c][1])
            for c in ordered[: top_k or self.config.top_k]
        ]
        best = ordered[0]
        confidence = scores[best]
        margin = confidence - (scores[ordered[1]] if len(ordered) > 1 else 0.0)
        unknown = confidence < self.config.min_confidence or (
            len(ordered) > 1 and margin < self.config.min_margin
        )
        return Prediction(
            None if unknown else best,
            confidence,
            parts[best][0],
            parts[best][1],
            margin,
            candidates,
            unknown,
        )

    def save(self, path: str) -> None:
        data = {
            "format_version": 1,
            "config": self.config.__dict__,
            "classes": self._classes,
            "examples": self._examples,
            "owners": self._owners,
            "embeddings": self._embeddings.tolist(),
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str, embedding_model=None) -> "Classifier":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format_version") != 1:
            raise ValueError("unsupported Semantra persistence format")
        instance = cls(embedding_model=embedding_model, config=ClassifierConfig(**data["config"]))
        instance._classes = {
            str(name): list(examples) for name, examples in data["classes"].items()
        }
        instance._examples = list(data["examples"])
        instance._owners = list(data["owners"])
        instance._bm25 = BM25(instance._examples)
        instance._embeddings = np.asarray(data["embeddings"], dtype=np.float32)
        if len(instance._examples) != len(instance._owners) or instance._embeddings.shape[0] != len(
            instance._examples
        ):
            raise ValueError("invalid classifier persistence data")
        return instance
