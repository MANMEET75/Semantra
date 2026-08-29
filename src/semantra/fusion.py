class WeightedScoreFusion:
    def __init__(self, semantic_weight: float = 0.75, lexical_weight: float = 0.25):
        total = semantic_weight + lexical_weight
        if total <= 0:
            raise ValueError("at least one score weight must be positive")
        self.semantic_weight = semantic_weight / total
        self.lexical_weight = lexical_weight / total

    def combine(self, semantic: float, lexical: float) -> float:
        return self.semantic_weight * semantic + self.lexical_weight * lexical
