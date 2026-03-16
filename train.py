import argparse
import json
import logging
import pickle
import time
from pathlib import Path
import numpy as np
import yaml
from rich.console import Console
from rich.table import Table

from src.data.loader import load_smd, split_machines
from src.data.windows import make_windows_for_machines
from src.evaluation.metrics import compute_metrics
from src.evaluation.threshold import find_best_threshold
from src.features.engineer import FeaturePipeline
from src.models.baseline import MajorityClassBaseline, PersistenceBaseline
from src.models.classical import build_logistic_regression, build_random_forest, tune_xgboost
from src.models.lstm import predict_lstm, train_lstm

console = Console()


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def _save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _section(title: str) -> None:
    console.rule(f"[bold cyan]{title}")


def _elapsed(t0: float) -> str:
    return f"({time.time() - t0:.1f}s)"


def _eval_classical(model, X_feat, y, threshold):
    scores = model.predict_proba(X_feat)[:, 1]
    return compute_metrics(y, scores, threshold), scores


def _eval_lstm(model, X_raw, y, norm, threshold):
    mean_, std_ = norm
    scores = predict_lstm(model, X_raw, mean_, std_)
    return compute_metrics(y, scores, threshold), scores


def _print_table(results: dict) -> None:
    tbl = Table(title="Model Comparison", show_lines=True, header_style="bold magenta")
    tbl.add_column("Model",           style="cyan", no_wrap=True)
    tbl.add_column("Split",           justify="center")
    tbl.add_column("AUROC",           justify="right")
    tbl.add_column("AUPRC",        justify="right", style="yellow")
    tbl.add_column("F1",              justify="right")
    tbl.add_column("Precision",       justify="right")
    tbl.add_column("Recall",          justify="right")
    tbl.add_column("Threshold",       justify="right")

    for name, r in results.items():
        for split in ("val", "test"):
            m = r[split]
            tbl.add_row(
                name if split == "val" else "",
                split,
                f"{m['auroc']:.3f}",
                f"{m['auprc']:.3f}",
                f"{m['f1']:.3f}",
                f"{m['precision']:.3f}",
                f"{m['recall']:.3f}",
                f"{r['threshold']:.3f}",
            )

    console.print(tbl)
    console.print(
        "\n[dim] AUPRC is the primary metric for imbalanced binary classification.\n"
        "  Threshold is tuned on the validation set and held fixed for test evaluation.[/dim]"
    )


def main(cfg: dict, selected_machines=None, fast: bool = False) -> None:
    out_dir = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    W    = cfg["window"]["W"]
    H    = cfg["window"]["H"]
    step = cfg["window"]["step"]

    _section("1 · Loading Data")
    machines = load_smd(
        cfg["paths"]["raw_dir"],
        machines=selected_machines or cfg["data"].get("machines"),
    )
    console.print(f"  Machines loaded : [bold]{len(machines)}[/bold]")
    console.print(f"  Features per machine : {machines[0].n_features}")

    train_ms, val_ms, test_ms = split_machines(
        machines,
        val_frac=cfg["split"]["val_frac"],
        test_frac=cfg["split"]["test_frac"],
        seed=cfg["split"]["seed"],
    )
    console.print(
        f"  Split  →  train: {len(train_ms)}  |  val: {len(val_ms)}  |  test: {len(test_ms)}"
    )

    _section(f"2 · Sliding Windows  (W={W}, H={H}, step={step})")
    X_tr_raw, y_tr, _  = make_windows_for_machines(train_ms, W, H, step)
    X_val_raw, y_val, _ = make_windows_for_machines(val_ms,   W, H, step)
    X_te_raw, y_te, _  = make_windows_for_machines(test_ms,  W, H, step)

    for tag, X, y in (("train", X_tr_raw, y_tr), ("val", X_val_raw, y_val), ("test", X_te_raw, y_te)):
        console.print(
            f"  {tag:5s}  {len(y):8,d} windows  "
            f"pos={y.sum():6,d}  ({y.mean():.2%} incident rate)"
        )

    _section("3 · Feature Engineering")
    feat_pipe = FeaturePipeline(sub_windows=cfg["features"]["sub_windows"])
    X_tr_feat  = feat_pipe.fit_transform(X_tr_raw)
    X_val_feat = feat_pipe.transform(X_val_raw)
    X_te_feat  = feat_pipe.transform(X_te_raw)
    console.print(f"  Feature matrix : {X_tr_feat.shape[1]} features per window")
    _save(feat_pipe, out_dir / "feature_pipeline.pkl")

    results: dict = {} 

    _section("4 · Training — Baselines")
    for name, model, use_raw in (
        ("majority_class",  MajorityClassBaseline(),        False),
        ("persistence",     PersistenceBaseline(k=5),       True),
    ):
        X_fit  = X_tr_raw  if use_raw else X_tr_feat
        X_valf = X_val_raw if use_raw else X_val_feat
        X_tef  = X_te_raw  if use_raw else X_te_feat
        model.fit(X_fit, y_tr)
        val_scores  = model.predict_proba(X_valf)[:, 1]
        thresh, _   = find_best_threshold(y_val, val_scores)
        test_scores = model.predict_proba(X_tef)[:, 1]
        results[name] = dict(
            val       = compute_metrics(y_val, val_scores,  thresh),
            test      = compute_metrics(y_te,  test_scores, thresh),
            threshold = thresh,
        )
        console.print(f"  [green]✓[/green] {name}  val_auprc={results[name]['val']['auprc']:.4f}")

    _section("5 · Training — Logistic Regression")
    t0 = time.time()
    lr_model = build_logistic_regression(C=cfg["models"]["logistic_regression"]["C"])
    lr_model.fit(X_tr_feat, y_tr)
    val_scores = lr_model.predict_proba(X_val_feat)[:, 1]
    thresh, _  = find_best_threshold(y_val, val_scores)
    test_scores = lr_model.predict_proba(X_te_feat)[:, 1]
    results["logistic_regression"] = dict(
        val       = compute_metrics(y_val, val_scores,  thresh),
        test      = compute_metrics(y_te,  test_scores, thresh),
        threshold = thresh,
    )
    console.print(
        f"  [green]✓[/green] logistic_regression  "
        f"val_auprc={results['logistic_regression']['val']['auprc']:.4f}  {_elapsed(t0)}"
    )
    _save(lr_model, out_dir / "lr_model.pkl")

    _section("6 · Training — Random Forest")
    t0 = time.time()
    rf_cfg   = cfg["models"]["random_forest"]
    rf_model = build_random_forest(
        n_estimators    = rf_cfg["n_estimators"],
        max_depth       = rf_cfg["max_depth"],
        min_samples_leaf= rf_cfg["min_samples_leaf"],
    )
    rf_model.fit(X_tr_feat, y_tr)
    val_scores  = rf_model.predict_proba(X_val_feat)[:, 1]
    thresh, _   = find_best_threshold(y_val, val_scores)
    test_scores = rf_model.predict_proba(X_te_feat)[:, 1]
    results["random_forest"] = dict(
        val       = compute_metrics(y_val, val_scores,  thresh),
        test      = compute_metrics(y_te,  test_scores, thresh),
        threshold = thresh,
    )
    console.print(
        f"  [green]✓[/green] random_forest  "
        f"val_auprc={results['random_forest']['val']['auprc']:.4f}  {_elapsed(t0)}"
    )
    _save(rf_model, out_dir / "rf_model.pkl")

    _section("7 · Training — XGBoost + Optuna")
    t0 = time.time()
    n_trials = 10 if fast else cfg["models"]["xgboost"]["n_optuna_trials"]
    console.print(f"  Running {n_trials} Optuna trials…")
    xgb_model, best_xgb_params = tune_xgboost(
        X_tr_feat, y_tr, X_val_feat, y_val, n_trials=n_trials
    )
    val_scores  = xgb_model.predict_proba(X_val_feat)[:, 1]
    thresh, _   = find_best_threshold(y_val, val_scores)
    test_scores = xgb_model.predict_proba(X_te_feat)[:, 1]
    results["xgboost"] = dict(
        val        = compute_metrics(y_val, val_scores,  thresh),
        test       = compute_metrics(y_te,  test_scores, thresh),
        threshold  = thresh,
        best_params= best_xgb_params,
    )
    console.print(
        f"  [green]✓[/green] xgboost  "
        f"val_auprc={results['xgboost']['val']['auprc']:.4f}  {_elapsed(t0)}"
    )
    _save(xgb_model, out_dir / "xgb_model.pkl")

    _section("8 · Training — Bidirectional LSTM")
    t0        = time.time()
    lstm_cfg  = cfg["models"]["lstm"]
    lstm_epochs = 15 if fast else lstm_cfg["epochs"]
    lstm_model, mean_, std_ = train_lstm(
        X_tr_raw,  y_tr,
        X_val_raw, y_val,
        hidden_dim      = lstm_cfg["hidden_dim"],
        num_layers      = lstm_cfg["num_layers"],
        dropout         = lstm_cfg["dropout"],
        batch_size      = lstm_cfg["batch_size"],
        epochs          = lstm_epochs,
        lr              = lstm_cfg["lr"],
        patience        = lstm_cfg["patience"],
        checkpoint_path = out_dir / "lstm_best.pt",
    )
    val_scores  = predict_lstm(lstm_model, X_val_raw, mean_, std_)
    thresh, _   = find_best_threshold(y_val, val_scores)
    test_scores = predict_lstm(lstm_model, X_te_raw,  mean_, std_)
    results["lstm"] = dict(
        val       = compute_metrics(y_val, val_scores,  thresh),
        test      = compute_metrics(y_te,  test_scores, thresh),
        threshold = thresh,
        norm      = (mean_, std_),
    )
    console.print(
        f"  [green]✓[/green] lstm  "
        f"val_auprc={results['lstm']['val']['auprc']:.4f}  {_elapsed(t0)}"
    )
    _save((mean_, std_), out_dir / "lstm_norm.pkl")

    _section("Results")
    _print_table(results)

    results_json = {
        name: {
            "val":       {k: round(v, 4) for k, v in r["val"].items()   if isinstance(v, float)},
            "test":      {k: round(v, 4) for k, v in r["test"].items()  if isinstance(v, float)},
            "threshold": round(r["threshold"], 4),
        }
        for name, r in results.items()
    }
    json_path = out_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    console.print(f"\n  [dim]Full results → {json_path}[/dim]")
    console.print(
        f"  [dim]Run  python evaluate.py  for PR curves and feature importances.[/dim]"
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train all incident-prediction models")
    parser.add_argument("--config",    default="configs/config.yaml")
    parser.add_argument("--machines",  nargs="*", default=None,
                        help="Subset of machine names to use (e.g. machine-1-1 machine-1-2)")
    parser.add_argument("--fast",      action="store_true",
                        help="Quick run: 10 Optuna trials, 15 LSTM epochs")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    cfg = load_config(args.config)
    main(cfg, selected_machines=args.machines, fast=args.fast)
