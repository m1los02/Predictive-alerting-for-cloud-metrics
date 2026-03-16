import numpy as np


class MajorityClassBaseline:
    """Trivial baseline: always predicts class 0 (high acc beacuse dataset is imbalanced)"""

    pos_rate_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MajorityClassBaseline":
        self.pos_rate_ = float(y.mean())
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X), dtype=np.int8)

    def predict_proba(self, X: np.ndarray) -> np.ndarray: # for AUPRC
        scores = np.full(len(X), self.pos_rate_, dtype=np.float32)
        return np.stack([1.0 - scores, scores], axis=1)


class PersistenceBaseline:
    """
    Persistence baseline: score = max metric value in the last k steps
    Uses only the raw window 
    """

    def __init__(self, k: int = 5):
        """
        Args:
            k: Number of tail steps to look at
        """
        self.k = k
        self._global_max: float = 1.0
        self._global_min: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PersistenceBaseline":
        tail = X[:, -self.k:, :]
        self._global_max = float(tail.max())
        self._global_min = float(tail.min())
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.predict_proba(X)[:, 1]
        return (scores >= 0.5).astype(np.int8)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        tail   = X[:, -self.k:, :]
        scores = tail.max(axis=(1, 2))
        denom  = max(self._global_max - self._global_min, 1e-8)
        scores = (scores - self._global_min) / denom
        scores = np.clip(scores, 0.0, 1.0).astype(np.float32)
        return np.stack([1.0 - scores, scores], axis=1)
