from typing import List, Optional, Tuple
import numpy as np
from sklearn.preprocessing import StandardScaler


def _rolling_stats(X: np.ndarray, sub_windows: List[int]) -> np.ndarray:
    """Mean, std, min, max, median over sub-windows"""
    N, W, F = X.shape
    parts = []
    for sw in sub_windows:
        sw = min(sw, W)
        x_sub = X[:, -sw:, :]
        parts.append(x_sub.mean(axis=1))
        parts.append(x_sub.std(axis=1))
        parts.append(x_sub.min(axis=1))
        parts.append(x_sub.max(axis=1))
        parts.append(np.median(x_sub, axis=1))
    return np.concatenate(parts, axis=1)


def _diff_stats(X: np.ndarray) -> np.ndarray:
    """Statistics of the first-order differences x(t) - x(t-1)"""
    dX = np.diff(X, axis=1)
    return np.concatenate([
        dX.mean(axis=1),
        dX.std(axis=1),
        np.abs(dX).mean(axis=1),
    ], axis=1)


def _trend_slope(X: np.ndarray) -> np.ndarray:
    """Linear regression slope across the window for each feature"""
    N, W, F = X.shape
    t = np.arange(W, dtype=np.float32) - (W - 1) / 2.0 
    denom = (t * t).sum() + 1e-8
    return (X * t[None, :, None]).sum(axis=1) / denom


def _zero_crossing_rate(X: np.ndarray) -> np.ndarray:
    """Measures sign change (in neighbouring timestamps)"""
    dX = np.diff(X, axis=1)
    signs = np.sign(dX)
    crossings = (signs[:, 1:, :] * signs[:, :-1, :] < 0)
    return crossings.mean(axis=1).astype(np.float32)


def _peak_fraction(X: np.ndarray, percentile: float = 90.0) -> np.ndarray:
    """Fraction of timesteps in the window that exceed the defined percentile value of that window"""
    thresholds = np.percentile(X, percentile, axis=1, keepdims=True)
    return (X > thresholds).mean(axis=1).astype(np.float32)


def extract_features(
    X: np.ndarray,
    sub_windows: Optional[List[int]] = None,
) -> np.ndarray:
    """Extract handcrafted features from windows"""
    N, W, F = X.shape
    if sub_windows is None:
        sub_windows = sorted({max(1, W // 4), max(2, W // 2), W})

    feature_blocks = [
        _rolling_stats(X, sub_windows),  
        _diff_stats(X),                    
        _trend_slope(X),                  
        _zero_crossing_rate(X),           
        _peak_fraction(X),
    ]
    return np.concatenate(feature_blocks, axis=1).astype(np.float32)


def get_feature_names(
    W: int,
    F: int,
    sub_windows: Optional[List[int]] = None,
) -> List[str]:
    """Return feature names for each extracted feature"""
    if sub_windows is None:
        sub_windows = sorted({max(1, W // 4), max(2, W // 2), W})

    names: List[str] = []
    for sw in sub_windows:
        for stat in ("mean", "std", "min", "max", "median"):
            for f in range(F):
                names.append(f"roll{sw}_{stat}_f{f}")
    for stat in ("mean_diff", "std_diff", "absmean_diff"):
        for f in range(F):
            names.append(f"{stat}_f{f}")
    for f in range(F):
        names.append(f"trend_slope_f{f}")
    for f in range(F):
        names.append(f"zcr_f{f}")
    for f in range(F):
        names.append(f"peak_frac_f{f}")
    return names


class FeaturePipeline:
    def __init__(self, sub_windows=None):
        self.sub_windows = sub_windows
        self.scaler      = StandardScaler()

    def fit_transform(self, X: np.ndarray, batch_size: int = 50_000) -> np.ndarray:
        feats = self._extract_batched(X, batch_size)
        return self.scaler.fit_transform(feats).astype(np.float32)

    def transform(self, X: np.ndarray, batch_size: int = 50_000) -> np.ndarray:
        feats = self._extract_batched(X, batch_size)
        return self.scaler.transform(feats).astype(np.float32)

    def _extract_batched(self, X: np.ndarray, batch_size: int) -> np.ndarray:
        parts = []
        for start in range(0, len(X), batch_size):
            parts.append(extract_features(X[start : start + batch_size], self.sub_windows))
        return np.concatenate(parts, axis=0)

    @property
    def n_features_(self) -> int:
        return self.scaler.n_features_in_