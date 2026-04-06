#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pl_fit_on_tgs_no_mit.py — Pliable Lasso on TGS with Z-set search across categories
(no resampling; hierarchy preserved; no leakage; NO max_interaction_terms restriction)
"""

import json, os, re
from time import time
from typing import Optional, Sequence, Tuple, List, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, f1_score, precision_recall_curve
)
from sklearn.linear_model import LassoCV
from plasso import PliableLasso

# ===================
# CONFIG
# ===================
CONFIG = {
    "BALANCED_PATH":   "C:/Users/enthe/Desktop/Thesis/data/tgs_data/tgs_dataset_normalized_balanced.csv",
    "outer_folds": 5,
    "inner_folds": 3,
    "alpha_grid": [0.01, 0.05,0.1, 0.3, 0.5, 0.7],
    "cv_grid":  [0.05, 0.10, 0.20, 0.30, 0.40],
    # "mit_grid": [5, 10, 20, 50, 100],  # REMOVED
    "cv_grid_fallback":  [0.30, 0.40, 0.50],
    # "mit_grid_fallback": [3, 5, 10],   # REMOVED
    "primary_metric": "average_precision",
    "threshold_strategy": "maximize_f1",
    "fixed_threshold": 0.5,
    "z_k": 15,
    "z_max_abs_corr": 0.40,
    "z_categories": ["random", "high_variance"],
    "z_n_per_category": 8,
    "z_jitter": 1e-3,
    "z_random_state": 13,
    "z_max_abs_corr_relaxed": 0.70,
    "MIN_VAL_SAMPLES": 20,
    "MAX_CV_FRACTION": 0.5,
    "CV_STEP_UP": 0.05,
    "MIN_VAL_POS": 2,
    "MIN_VAL_NEG": 2,
    "SAVE_PRED": True,
    "SAVE_COEF_CSV": True,
}

RANDOM_STATE_INNER = 42
RANDOM_STATE_OUTER = 123

def read_tgs_csv(path: str) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    if "MCI" not in df.columns:
        raise ValueError("Expected a column named 'MCI' as the target.")
    y = pd.to_numeric(df["MCI"], errors="coerce").to_numpy(dtype=float)
    x_cols = [c for c in df.columns if c != "MCI" and pd.api.types.is_numeric_dtype(df[c])]
    X = df[x_cols].apply(pd.to_numeric, errors="coerce")
    if X.isnull().any().any() or np.isnan(y).any():
        raise ValueError("Non-numeric or NaN values found in X/y. Please clean/impute first.")
    return X, y.astype(int), x_cols

def stem_for(tag: str) -> str:
    base = re.sub(r'[^A-Za-z0-9_.-]+', '_', tag)
    return f"tgs_{base}"

def _mean_no_warn(arr_like) -> float:
    arr = np.asarray(arr_like, dtype=float)
    m = np.isfinite(arr)
    return float(arr[m].mean()) if m.any() else float("nan")

def _best_threshold_f1(y_true, scores) -> float:
    pr, rc, th = precision_recall_curve(y_true, scores)
    f1s, ts = [], []
    for p, r, t in zip(pr[1:], rc[1:], th):
        f1 = 2*p*r/(p+r) if (p+r) else 0.0
        f1s.append(f1); ts.append(t)
    return float(ts[int(np.argmax(f1s))]) if ts else 0.5

def _kfold_with_min_class(y: np.ndarray, desired_splits: int, seed: int) -> StratifiedKFold:
    pos = int(np.sum(y)); neg = int(len(y) - pos)
    max_splits = max(2, min(pos, neg))
    n_splits = max(2, min(desired_splits, max_splits))
    if n_splits < desired_splits:
        print(f"[INFO] Reducing splits {desired_splits}→{n_splits} to keep both classes in every fold "
              f"(pos={pos}, neg={neg}).")
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

def _corr_matrix(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(0, keepdims=True)
    C = np.corrcoef(Xc, rowvar=False)
    return np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)

def _topk_from_order(order: np.ndarray, k: int) -> np.ndarray:
    k = max(1, min(int(k), len(order)))
    return np.asarray(order[:k], dtype=int)

def _pruned_from_order(order: np.ndarray, C: np.ndarray, k: int, max_abs_corr: float) -> np.ndarray:
    chosen: List[int] = []
    for j in order:
        if len(chosen) >= k:
            break
        if all(abs(C[j, i]) <= max_abs_corr for i in chosen):
            chosen.append(int(j))
    if len(chosen) < k:
        for j in order:
            if j not in chosen:
                chosen.append(int(j))
                if len(chosen) >= k:
                    break
    return np.asarray(chosen[:k], dtype=int)

def _build_z_candidates_categories(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    max_abs_corr: float,
    categories: Sequence[str],
    n_per_category: int,
    jitter: float,
    seed: int
) -> List[Tuple[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    p = X.shape[1]
    C = _corr_matrix(X); np.fill_diagonal(C, 0.0)
    avg_abs = np.mean(np.abs(C), axis=0)
    vari = np.var(X, axis=0, ddof=1)

    candidates: List[Tuple[str, np.ndarray]] = []
    for cat in categories:
        for _ in range(int(n_per_category)):
            if cat == "random":
                order = rng.permutation(p)
                z_idx = _topk_from_order(order, k)
            elif cat == "high_variance":
                score = -(vari + rng.normal(0.0, jitter, size=p))
                order = np.argsort(score)
                z_idx = _topk_from_order(order, k)
            else:
                raise ValueError(f"Unknown Z category: {cat}")
            candidates.append((cat, z_idx))

    seen = set(); uniq: List[Tuple[str, np.ndarray]] = []
    for cat, z in candidates:
        key = (cat, tuple(sorted(map(int, z))))
        if key not in seen:
            seen.add(key)
            uniq.append((cat, z))
    return uniq

def get_beta(model) -> Optional[np.ndarray]:
    for name in ("beta", "beta_", "coef_", "coef"):
        if hasattr(model, name):
            val = getattr(model, name)
            if val is not None:
                return np.asarray(val).reshape(-1)
    return None

def get_theta(model) -> Optional[np.ndarray]:
    for name in ("theta", "theta_", "Theta_", "Theta"):
        if hasattr(model, name):
            val = getattr(model, name)
            if val is not None:
                arr = np.asarray(val)
                if arr.ndim == 2:
                    return arr
    return None

def get_intercept(model) -> float:
    for name in ("intercept_", "intercept", "beta0_", "beta0"):
        if hasattr(model, name):
            try:
                v = getattr(model, name)
                return float(np.asarray(v).ravel()[0])
            except Exception:
                pass
    return 0.0

def _effective_cv_frac(cv_frac: float, n_train: int, ytr: np.ndarray) -> float:
    n_train = max(1, int(n_train))
    pi = float(np.clip(ytr.mean(), 1e-9, 1 - 1e-9))

    need_total = CONFIG["MIN_VAL_SAMPLES"] / n_train
    need_pos   = CONFIG["MIN_VAL_POS"]   / (n_train * pi)
    need_neg   = CONFIG["MIN_VAL_NEG"]   / (n_train * (1.0 - pi))

    eff = max(cv_frac, need_total, need_pos, need_neg) + 1e-6
    return float(min(eff, CONFIG["MAX_CV_FRACTION"]))

def _fit_plasso_safe(Xtr: np.ndarray, ytr: np.ndarray,
                     z_idx: np.ndarray, cv_init: float,
                     alpha: float = 0.1,
                     max_tries: int = 8):
    ntr = Xtr.shape[0]
    cv_used = _effective_cv_frac(cv_init, ntr, ytr)
    err = None

    for _ in range(max_tries):
        try:
            mdl = PliableLasso(cv=cv_used,max_interaction_terms=100, verbose=False, eps=1e-4, normalize=True, alpha=alpha)
            mdl.fit(Xtr, Xtr[:, z_idx], ytr)
            return mdl, cv_used, None
        except Exception as e:
            err_msg = str(e)
            err = err_msg
            next_cv = min(cv_used + CONFIG["CV_STEP_UP"], CONFIG["MAX_CV_FRACTION"])
            cv_used = next_cv
            if cv_used >= CONFIG["MAX_CV_FRACTION"] - 1e-12:
                break

    return None, cv_used, err

def inner_cv_grid(X: np.ndarray, y: np.ndarray, z_idx: np.ndarray,
                  alpha_list: Sequence[float], cv_list: Sequence[float],
                  metric: str):
    rows = []
    best_score, best_alpha, best_cv = -np.inf, None, None
    skf = _kfold_with_min_class(y, CONFIG["inner_folds"], RANDOM_STATE_INNER)

    for alpha in alpha_list:
        for cv in cv_list:
            scores, tried_cvs, fails = [], [], 0
            for tr, va in skf.split(X, y):
                mdl, cv_used, err = _fit_plasso_safe(X[tr], y[tr], z_idx, cv_init=cv, alpha=alpha)
                if mdl is not None:
                    mdl.alpha = alpha
                    mdl.cv = cv_used
                tried_cvs.append(cv_used)
                if mdl is None:
                    fails += 1
                    continue
                scores_va = np.asarray(mdl.predict(X[va], X[va][:, z_idx])).ravel()
                try:
                    if metric == "average_precision":
                        s = average_precision_score(y[va], scores_va)
                    elif metric == "roc_auc":
                        s = roc_auc_score(y[va], scores_va)
                    else:
                        raise ValueError(f"Unsupported metric: {metric}")
                except Exception:
                    s = np.nan
                scores.append(s)
            m = _mean_no_warn(scores)
            s = float(np.nanstd(scores)) if len(scores) else float("nan")
            rows.append({
                "alpha": alpha,
                "cv_requested": cv,
                "cv_mean_used": float(np.mean(tried_cvs)) if tried_cvs else np.nan,
                f"mean_{metric}": m,
                f"std_{metric}": s,
                "fold_failures": int(fails),
            })
            if np.isfinite(m) and (m > best_score):
                best_score, best_alpha, best_cv = m, alpha, cv

    leaderboard = pd.DataFrame(rows).sort_values(by=f"mean_{metric}", ascending=False).reset_index(drop=True)
    if best_alpha is None or best_cv is None:
        raise RuntimeError("All (alpha,cv) combos failed to fit for this Z candidate.")
    return best_alpha, best_cv, best_score, leaderboard

def choose_best_z_via_inner_cv_categories(
    X: np.ndarray,
    y: np.ndarray,
    x_cols: List[str],
    k: int,
    max_abs_corr: float,
    categories: Sequence[str],
    n_per_category: int,
    jitter: float,
    seed: int,
    alpha_list: Sequence[float],
    cv_list: Sequence[float],
    metric: str
):
    cands = _build_z_candidates_categories(
        X, y, k=k, max_abs_corr=max_abs_corr,
        categories=categories, n_per_category=n_per_category,
        jitter=jitter, seed=seed
    )
    rows = []
    best = (None, None, None, -np.inf, None, None)
    for cid, (cat, z_idx) in enumerate(cands, 1):
        try:
            alpha_star, cv_star, score_star, _ = inner_cv_grid(
                X, y, z_idx, alpha_list=alpha_list, cv_list=cv_list, metric=metric
            )
        except Exception as e:
            rows.append({"candidate_id": cid, "category": cat, "z_size": int(len(z_idx)),
                         "best_score": np.nan, "error": str(e),
                         "z_names": ";".join([x_cols[int(j)] for j in z_idx])})
            continue
        rows.append({"candidate_id": cid, "category": cat, "z_size": int(len(z_idx)),
                     "best_score": float(score_star),
                     "best_alpha": float(alpha_star), "best_cv": float(cv_star),
                     "z_names": ";".join([x_cols[int(j)] for j in z_idx])})
        if score_star > best[3]:
            best = (z_idx, alpha_star, cv_star, score_star, cid, cat)

    z_idx_best, alpha_best, cv_best, score_best, cid_best, cat_best = best
    z_df = pd.DataFrame(rows).sort_values(by=["best_score"], ascending=False).reset_index(drop=True)
    if z_idx_best is None:
        raise RuntimeError("All Z candidates failed to fit. Consider reducing z_k or loosening z_max_abs_corr.")
    return z_idx_best, alpha_best, cv_best, score_best, cid_best, cat_best, z_df

def fit_one(dataset_tag: str, path: str):
    Xdf, y, x_cols = read_tgs_csv(path)
    X = Xdf.to_numpy(dtype=float)
    p = X.shape[1]

    # --- Single Lasso fit on all data for Z selection ---
    lasso = LassoCV(alphas=np.linspace(0.0001, 0.05, 30), cv=5, random_state=CONFIG["z_random_state"])
    lasso.fit(X, y)
    coefs = np.abs(lasso.coef_)
    main_indices = np.where(coefs > 1e-8)[0]

    # Select up to 15 mains with largest absolute coefficients
    if len(main_indices) > 15:
        sorted_idx = main_indices[np.argsort(-coefs[main_indices])]
        main_indices = sorted_idx[:15]

    selected_mains = [x_cols[j] for j in main_indices]
    print(f"Selected main effect indices (max 15): {main_indices}")

    z_idx = main_indices
    chosen_z_names = [x_cols[j] for j in z_idx]
    chosen_z_category = "lasso_selected"

    # --- Hyperparameter tuning on all data ---
    best_alpha, best_cv, _, _ = inner_cv_grid(
        X, y, z_idx,
        alpha_list=CONFIG["alpha_grid"],
        cv_list=CONFIG["cv_grid"],
        metric=CONFIG["primary_metric"]
    )

    # --- Fit PliableLasso on all data ---
    t0 = time()
    mdl, cv_used, err = _fit_plasso_safe(
        X, y, z_idx,
        cv_init=best_cv,
        alpha=best_alpha
    )
    fit_time = time() - t0

    if mdl is None:
        print(f"[ERROR] PliableLasso fit failed: {err}")
        summary = {
            "dataset": os.path.basename(path),
            "selected_hyperparams": {
                "alpha": float(best_alpha),
                "cv_requested": float(best_cv)
            },
            "chosen_z_category": chosen_z_category,
            "chosen_z_names": chosen_z_names,
            "selected_mains": selected_mains,
            "selected_interactions": [],
            "intercept": None,
            "fit_time_seconds": fit_time,
            "mse": None,
            "accuracy": None,
            "error": err
        }
        return summary

    # --- Extract coefficients and interactions ---
    beta = get_beta(mdl)
    theta = get_theta(mdl)
    intercept = get_intercept(mdl)

    selected_mains_pl = []
    if beta is not None:
        for j in range(len(beta)):
            if abs(beta[j]) > 1e-8:
                selected_mains_pl.append(x_cols[j])

    selected_interactions = []
    if theta is not None:
        for i in range(theta.shape[0]):
            for j in range(theta.shape[1]):
                if abs(theta[i, j]) > 1e-8:
                    selected_interactions.append([x_cols[i], x_cols[z_idx[j]]])

    # --- Predict and calculate metrics ---
    y_pred = mdl.predict(X, X[:, z_idx]).ravel()
    mse = float(np.mean((y - y_pred) ** 2))
    accuracy = float(np.mean((y == (y_pred > 0.5).astype(int))))

    # --- Prepare output dict ---
    summary = {
        "dataset": os.path.basename(path),
        "selected_hyperparams": {
            "alpha": float(best_alpha),
            "cv_requested": float(best_cv)
        },
        "chosen_z_category": chosen_z_category,
        "chosen_z_names": chosen_z_names,
        "selected_mains": selected_mains,
        "selected_mains_pl": selected_mains_pl, 
        "selected_interactions": selected_interactions,
        "intercept": float(intercept),
        "fit_time_seconds": fit_time,
        "mse": mse,
        "accuracy": accuracy
    }
    return summary

def main():
    path = CONFIG["BALANCED_PATH"]
    tag = "balanced"

    if not os.path.exists(path):
        print(f"[WARN] {path} not found; exiting.")
        return
    
    summary = fit_one(tag, path)

    dataset_name = os.path.basename(path).replace(".csv", "")
    out_dir = "results/tgs_results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"selected_tgs_{tag}_md.json")

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved results to {out_path}")

    # Save selected interactions with main/modifier specification
    selected_interactions = summary.get("selected_interactions", [])
    selected_interactions_mm = []
    for pair in selected_interactions:
        selected_interactions_mm.append({
            "main_effect": pair[0],
            "modifier": pair[1]
        })

    out_path_mm = os.path.join(out_dir, f"selected_tgs_{tag}_main_modifier.json")
    with open(out_path_mm, "w") as f:
        json.dump(selected_interactions_mm, f, indent=2)
    print(f"Saved main/modifier interactions to {out_path_mm}")
    

if __name__ == "__main__":
    main()