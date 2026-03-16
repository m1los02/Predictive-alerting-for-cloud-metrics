from typing import Tuple
import numpy as np
from sklearn.metrics import precision_recall_curve


def find_best_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: str = "f1",
) -> Tuple[float, float]:
    """
    Sweep all candidate thresholds and return the one that maximises the chosen metric on the provided data

    Args:
        y_true:  Ground-truth binary labels.
        y_score: Predicted probability scores.
        metric:  One of:
                 - f1: standard F1  (β = 1, balanced)
                 - f1_beta_2: recall-heavy (β = 2, catches more incidents)
                 - f1_beta_0.5: precision-heavy (β = 0.5, fewer false alarms)

    Returns:
        (best_threshold, best_metric_value)
    """
    beta_map = {"f1": 1.0, "f1_beta_2": 2.0, "f1_beta_0.5": 0.5}
    if metric not in beta_map:
        raise ValueError(
            f"Unknown metric '{metric}'. "
            f"Choose from: {list(beta_map.keys())}"
        )
    beta  = beta_map[metric]
    beta2 = beta ** 2

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)

    with np.errstate(divide="ignore", invalid="ignore"):
        f_scores = np.where(
            (beta2 * precisions[:-1] + recalls[:-1]) > 0,
            (1 + beta2)
            * precisions[:-1]
            * recalls[:-1]
            / (beta2 * precisions[:-1] + recalls[:-1]),
            0.0,
        )

    best_idx = int(np.argmax(f_scores))
    return float(thresholds[best_idx]), float(f_scores[best_idx])


def threshold_sweep(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_points: int = 200,
) -> dict:
    """Compute precision, recall, and F1 across a grid of thresholds"""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1_scores = np.where(
            precisions[:-1] + recalls[:-1] > 0,
            2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1]),
            0.0,
        )
    return {
        "thresholds":  thresholds,
        "precisions":  precisions[:-1],
        "recalls":     recalls[:-1],
        "f1_scores":   f1_scores,
    }
