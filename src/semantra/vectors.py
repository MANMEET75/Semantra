import numpy as np


class NumpyVectorSearcher:
    def similarities(self, query: np.ndarray, documents: np.ndarray) -> np.ndarray:
        denom = np.linalg.norm(documents, axis=1) * np.linalg.norm(query)
        raw = np.divide(documents @ query, denom, out=np.zeros(len(documents)), where=denom != 0)
        return np.clip((raw + 1) / 2, 0, 1)
