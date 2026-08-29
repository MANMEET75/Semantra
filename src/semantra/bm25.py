import math
from collections import Counter
from typing import List, Sequence

from .preprocessing import tokenize


class BM25:
    def __init__(self, documents: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.documents = list(documents)
        self.k1, self.b = k1, b
        self.tokens = [tokenize(d) for d in self.documents]
        self.lengths = [len(t) for t in self.tokens]
        self.avgdl = sum(self.lengths) / len(self.lengths) if self.lengths else 1.0
        self.term_frequency = [Counter(t) for t in self.tokens]
        df = Counter(term for terms in self.tokens for term in set(terms))
        n = len(self.tokens)
        self.idf = {
            term: math.log(1 + (n - count + 0.5) / (count + 0.5)) for term, count in df.items()
        }

    def scores(self, query: str) -> List[float]:
        q = tokenize(query)
        values = []
        for terms, length, frequencies in zip(self.tokens, self.lengths, self.term_frequency):
            score = 0.0
            for term in q:
                if term not in frequencies:
                    continue
                f = frequencies[term]
                denom = f + self.k1 * (1 - self.b + self.b * length / self.avgdl)
                score += self.idf.get(term, 0.0) * f * (self.k1 + 1) / denom
            values.append(score)
        maximum = max(values, default=0.0)
        return [v / maximum if maximum else 0.0 for v in values]
