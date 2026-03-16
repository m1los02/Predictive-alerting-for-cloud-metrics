from typing import Optional, Tuple
import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
import xgboost as xgb
import torch

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _cuda_available() -> bool:
    return torch.cuda.is_available()
    

def _pos_weight(y: np.ndarray) -> float:
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    return float(n_neg) / max(1, n_pos)


def build_logistic_regression(C: float = 0.1) -> LogisticRegression:
    return LogisticRegression(
        C=C,
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
        multi_class="auto",
        random_state=42,
    )


def build_random_forest(
    n_estimators: int = 300,
    max_depth: Optional[int] = 15,
    min_samples_leaf: int = 5,
) -> RandomForestClassifier:
    
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )


def _xgb_trial(
    trial: optuna.Trial,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 700),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
    }
    model = xgb.XGBClassifier(
        **params,
        scale_pos_weight=_pos_weight(y_tr),
        eval_metric="aucpr",
        use_label_encoder=False,
        random_state=42,
        tree_method="hist",
        device="cuda" if _cuda_available() else "cpu",
        verbosity=0,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    y_score = model.predict_proba(X_val)[:, 1]
    return average_precision_score(y_val, y_score)


def tune_xgboost(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
) -> Tuple[xgb.XGBClassifier, dict]:
    """
    Run Optuna hyperparameter search and return the best model.
    After the search the model is retrained on the combined train + val data using the best parameters
    """
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(
        lambda t: _xgb_trial(t, X_tr, y_tr, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best_params = study.best_params

    # retrain
    X_all = np.concatenate([X_tr, X_val], axis=0)
    y_all = np.concatenate([y_tr, y_val], axis=0)

    best_model = xgb.XGBClassifier(
        **best_params,
        scale_pos_weight=_pos_weight(y_all),
        eval_metric="aucpr",
        use_label_encoder=False,
        random_state=42,
        tree_method="hist",
        device="cuda" if _cuda_available() else "cpu",
        verbosity=0,
    )
    best_model.fit(X_all, y_all)
    return best_model, best_params
