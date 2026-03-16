from typing import Dict, Optional
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    
    if len(np.unique(y_true)) < 2:
        return dict(
            auroc=float("nan"), auprc=float("nan"),
            f1=float("nan"), precision=float("nan"),
            recall=float("nan"), threshold=threshold,
        )

    y_pred = (y_score >= threshold).astype(np.int8)

    return dict(
        auroc=roc_auc_score(y_true, y_score),
        auprc=average_precision_score(y_true, y_score),
        f1=f1_score(y_true, y_pred, zero_division=0),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        threshold=float(threshold),
    )


def compute_lead_time(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> Optional[float]:
    """Compute mean alert lead time (in timesteps)"""
    y_pred = (y_score >= threshold).astype(np.int8)

    padded = np.concatenate([[0], y_true.astype(np.int8), [0]])
    diff   = np.diff(padded)
    onsets = np.where(diff == 1)[0]   # indices in y_true where an incident begins

    lead_times = []
    for onset in onsets:
        # consider alerts that fired before this onset
        pre_alerts = np.where(y_pred[:onset])[0]
        if len(pre_alerts) > 0:
            # earliest alert in the look-back period
            first_alert = pre_alerts[0]
            lead_times.append(onset - first_alert)

    return float(np.mean(lead_times)) if lead_times else None


def classification_report_dict(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    name: str = "model",
) -> Dict:
    """
    Full evaluation summary including lead-time 

    For the lead-time to be meaningful the inputs must be in temporal order,
    i.e. one machine's full test sequence rather than shuffled windows.
    """
    metrics = compute_metrics(y_true, y_score, threshold)
    lead    = compute_lead_time(y_true, y_score, threshold)
    metrics["lead_time_steps"] = lead if lead is not None else float("nan")
    metrics["model"]           = name
    return metrics
