import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import yaml
from sklearn.metrics import precision_recall_curve, roc_curve

from src.data.loader import load_smd, split_machines
from src.data.windows import make_windows_for_machines
from src.evaluation.metrics import compute_metrics, compute_lead_time
from src.evaluation.threshold import find_best_threshold, threshold_sweep
from src.features.engineer import FeaturePipeline, get_feature_names
from src.models.lstm import IncidentPredictor, predict_lstm

import argparse

plt.style.use("seaborn-v0_8-whitegrid")
COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]


def load_artifacts(out_dir: Path) -> dict:
    """Load all saved model artifacts from checkpoints directory."""
    import torch

    def _pkl(name):
        p = out_dir / name
        if not p.exists():
            return None
        with open(p, "rb") as f:
            return pickle.load(f)

    artifacts = {
        "feature_pipeline": _pkl("feature_pipeline.pkl"),
        "lr_model":         _pkl("lr_model.pkl"),
        "rf_model":         _pkl("rf_model.pkl"),
        "xgb_model":        _pkl("xgb_model.pkl"),
        "lstm_norm":        _pkl("lstm_norm.pkl"),
    }

    # load LSTM checkpoint
    ckpt_path = out_dir / "lstm_best.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        artifacts["lstm_ckpt"] = ckpt
    else:
        artifacts["lstm_ckpt"] = None

    # load results JSON
    res_path = out_dir / "results.json"
    if res_path.exists():
        with open(res_path) as f:
            artifacts["results_json"] = json.load(f)
    else:
        artifacts["results_json"] = {}

    return artifacts


def _pr_curves(model_scores: dict, y_true: np.ndarray, save_path: Path) -> None:
    """Plot Precision-Recall curves for all models"""
    fig, ax = plt.subplots(figsize=(8, 6))
    baseline_ap = y_true.mean()
    ax.axhline(baseline_ap, color="gray", ls="--", lw=1, label=f"Baseline (AP={baseline_ap:.3f})")

    for (name, scores), color in zip(model_scores.items(), COLORS):
        p, r, _ = precision_recall_curve(y_true, scores)
        from sklearn.metrics import average_precision_score
        ap = average_precision_score(y_true, scores)
        ax.plot(r, p, color=color, lw=2, label=f"{name}  (AP={ap:.3f})")

    ax.set_xlabel("Recall",    fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision–Recall Curves (Test Set)", fontsize=14)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _roc_curves(model_scores: dict, y_true: np.ndarray, save_path: Path) -> None:
    """Plot ROC curves for all models"""
    from sklearn.metrics import roc_auc_score
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=1, label="Random (AUC=0.500)")

    for (name, scores), color in zip(model_scores.items(), COLORS):
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = roc_auc_score(y_true, scores)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name}  (AUC={auc:.3f})")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curves (Test Set)", fontsize=14)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _threshold_sweep_plot(y_true: np.ndarray, scores: np.ndarray, save_path: Path, name: str = "XGBoost") -> None:
    """Plot Precision / Recall / F1 vs threshold for a single model"""
    sweep = threshold_sweep(y_true, scores)
    τ = sweep["thresholds"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(τ, sweep["precisions"], label="Precision", lw=2, color=COLORS[1])
    ax.plot(τ, sweep["recalls"],    label="Recall",    lw=2, color=COLORS[2])
    ax.plot(τ, sweep["f1_scores"],  label="F1",        lw=2, color=COLORS[3], ls="--")

    best_τ, best_f1 = find_best_threshold(y_true, scores)
    ax.axvline(best_τ, color="red", lw=1.5, ls=":", label=f"Best τ = {best_τ:.3f}")
    ax.scatter([best_τ], [best_f1], color="red", zorder=5, s=60)

    ax.set_xlabel("Decision Threshold τ", fontsize=12)
    ax.set_ylabel("Score",                fontsize=12)
    ax.set_title(f"{name}: Precision / Recall / F1 vs Threshold", fontsize=14)
    ax.legend(fontsize=10)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _feature_importance_plot(
    importances: np.ndarray,
    feature_names: list,
    save_path: Path,
    title: str,
    top_k: int = 30,
) -> None:
    """Horizontal bar chart of top-k feature importances"""
    idx = np.argsort(importances)[-top_k:]
    vals  = importances[idx]
    names = [feature_names[i] if i < len(feature_names) else f"feat_{i}" for i in idx]

    fig, ax = plt.subplots(figsize=(9, max(6, top_k * 0.28)))
    bars = ax.barh(range(top_k), vals, color=COLORS[1], alpha=0.8)
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Importance", fontsize=11)
    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _write_report(results_json: dict, save_path: Path) -> None:
    """Generate a markdown results table from saved JSON."""
    lines = [
        "# Evaluation Results\n",
        "## Model Comparison\n",
        "| Model | Split | AUROC | AUPRC ★ | F1 | Precision | Recall | Threshold |",
        "|-------|-------|------:|--------:|---:|----------:|-------:|----------:|",
    ]
    for name, r in results_json.items():
        for split in ("val", "test"):
            m = r[split]
            lines.append(
                f"| {name if split == 'val' else ''} | {split} "
                f"| {m.get('auroc', 'nan'):.3f} "
                f"| {m.get('auprc', 'nan'):.3f} "
                f"| {m.get('f1', 'nan'):.3f} "
                f"| {m.get('precision', 'nan'):.3f} "
                f"| {m.get('recall', 'nan'):.3f} "
                f"| {r['threshold']:.3f} |"
            )
    lines += [
        "",
        "> **AUPRC** (Average Precision) is the primary metric.",
        "> Baseline AUPRC equals the positive-class prevalence rate.",
        "",
        "## Plots",
        "- `plots/pr_curves.png` — Precision-Recall curves",
        "- `plots/roc_curves.png` — ROC curves",
        "- `plots/threshold_sweep.png` — Threshold sensitivity (XGBoost)",
        "- `plots/rf_feature_importance.png` — Random Forest top features",
        "- `plots/xgb_feature_importance.png` — XGBoost top features",
    ]
    save_path.write_text("\n".join(lines))


def main(cfg: dict) -> None:
    import torch

    out_dir     = Path(cfg["paths"]["output_dir"])
    results_dir = Path("results")
    plots_dir   = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    W    = cfg["window"]["W"]
    H    = cfg["window"]["H"]
    step = cfg["window"]["step"]

    machines = load_smd(
        cfg["paths"]["raw_dir"],
        machines=cfg["data"].get("machines"),
    )
    _, _, test_ms = split_machines(
        machines,
        val_frac=cfg["split"]["val_frac"],
        test_frac=cfg["split"]["test_frac"],
        seed=cfg["split"]["seed"],
    )
    X_te_raw, y_te, _ = make_windows_for_machines(test_ms, W, H, step)

    art = load_artifacts(out_dir)

    feat_pipe: FeaturePipeline = art["feature_pipeline"]
    X_te_feat = feat_pipe.transform(X_te_raw)

    model_scores = {}

    if art["lr_model"] is not None:
        model_scores["logistic_regression"] = art["lr_model"].predict_proba(X_te_feat)[:, 1]

    if art["rf_model"] is not None:
        model_scores["random_forest"] = art["rf_model"].predict_proba(X_te_feat)[:, 1]

    if art["xgb_model"] is not None:
        model_scores["xgboost"] = art["xgb_model"].predict_proba(X_te_feat)[:, 1]

    if art["lstm_ckpt"] is not None and art["lstm_norm"] is not None:
        ckpt     = art["lstm_ckpt"]
        mean_, std_ = art["lstm_norm"]
        n_feat   = X_te_raw.shape[2]
        model    = IncidentPredictor(n_feat)
        model.load_state_dict(ckpt["state_dict"])
        model_scores["lstm"] = predict_lstm(model, X_te_raw, mean_, std_)

    if not model_scores:
        print("No trained models found.  Run  python train.py  first.")
        return

    _pr_curves(model_scores,  y_te, plots_dir / "pr_curves.png")
    _roc_curves(model_scores, y_te, plots_dir / "roc_curves.png")

    if "xgboost" in model_scores:
        _threshold_sweep_plot(
            y_te, model_scores["xgboost"],
            plots_dir / "threshold_sweep.png",
            name="XGBoost",
        )

    n_feat_total = X_te_feat.shape[1]
    feat_names   = get_feature_names(W, X_te_raw.shape[2], cfg["features"]["sub_windows"])

    if art["rf_model"] is not None:
        _feature_importance_plot(
            art["rf_model"].feature_importances_,
            feat_names,
            plots_dir / "rf_feature_importance.png",
            "Random Forest — Top Feature Importances",
        )

    if art["xgb_model"] is not None:
        imp = art["xgb_model"].feature_importances_
        _feature_importance_plot(
            imp, feat_names,
            plots_dir / "xgb_feature_importance.png",
            "XGBoost — Top Feature Importances",
        )

    print("\n── Alert Lead Time (per-machine, test set) ──")
    for name, scores in model_scores.items():
        thresh = find_best_threshold(y_te, scores)[0]
        lt = compute_lead_time(y_te, scores, thresh)
        print(f"  {name:25s}  {f'{lt:.1f} steps' if lt else 'N/A':>15s}")

    if art["results_json"]:
        _write_report(art["results_json"], results_dir / "report.md")
        print(f"\nReport → results/report.md")

    print("Plots  → results/plots/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg)
