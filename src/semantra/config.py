from dataclasses import dataclass


@dataclass
class ClassifierConfig:
    semantic_weight: float = 0.75
    lexical_weight: float = 0.25
    min_confidence: float = 0.55
    min_margin: float = 0.05
    top_k: int = 3
    examples_per_class: int = 1000

    def __post_init__(self) -> None:
        if self.semantic_weight < 0 or self.lexical_weight < 0:
            raise ValueError("score weights must be non-negative")
        if self.semantic_weight + self.lexical_weight == 0:
            raise ValueError("at least one score weight must be positive")
        if not 0 <= self.min_confidence <= 1 or not 0 <= self.min_margin <= 1:
            raise ValueError("confidence and margin must be between 0 and 1")
        if self.top_k < 1 or self.examples_per_class < 1:
            raise ValueError("top_k and examples_per_class must be positive")
