#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pl_fit_on_tgs.py — Pliable Lasso on TGS with Z-set search across categories
(no resampling; hierarchy preserved; no leakage)
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

from plasso import PliableLasso

# ===================
# CONFIG
# ===================
CONFIG = {
    # Input files
    # Uncomment the one you want to use:
    "IMBALANCED_PATH": "C:/Users/enthe/Desktop/Thesis/data/tgs_data/tgs_dataset_normalized.csv",
    #"BALANCED_PATH":   "C:/Users/enthe/Desktop/Thesis/data/tgs_data/tgs_dataset_normalized_balanced.csv",

    # Nested CV (auto-reduce splits if a fold would miss a class)
    "outer_folds": 5,
    "inner_folds": 3,
    "alpha_grid": [0.05, 0.1, 0.3, 0.5, 0.7],

    # Hyperparam grid (primary)
    "cv_grid":  [0.05,0.10, 0.20, 0.30, 0.40],
    "mit_grid": [5, 10, 20, 50, 100],

    # Hyperparam grid (fallback used only if all Z candidates fail)
    "cv_grid_fallback":  [0.30, 0.40, 0.50],
    "mit_grid_fallback": [3, 5, 10],

    # Metric & thresholding
    "primary_metric": "average_precision",   # PR-AUC for selection
    "threshold_strategy": "maximize_f1",     # for reporting on outer test
    "fixed_threshold": 0.5,

    # ===== Z search settings =====
    "z_k": 10,                   # number of modifiers in each Z
    "z_max_abs_corr": 0.40,      # cap pairwise |corr| among modifiers (ONLY for 'low_corr')
    "z_categories": ["random", "high_variance"],  # <-- removed "low_corr"
    "z_n_per_category": 8,       # candidates per category per outer fold
    "z_jitter": 1e-3,            # tiny jitter to break ties
    "z_random_state": 13,        # RNG seed

    # Relaxed Z (used only on catastrophic failure)
    "z_max_abs_corr_relaxed": 0.70,

    # Internal CV guardrails for plasso (its own holdout)
    "MIN_VAL_SAMPLES": 20,
    "MAX_CV_FRACTION": 0.5,
    "CV_STEP_UP": 0.05,
    "MIN_VAL_POS": 2,
    "MIN_VAL_NEG": 2,

    # Saving toggles
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
                     z_idx: np.ndarray, cv_init: float, mit: int,
                     alpha: float = 0.1,
                     max_tries: int = 8):
    ntr = Xtr.shape[0]
    cv_used = _effective_cv_frac(cv_init, ntr, ytr)
    err = None
    tried_smaller_mit = False

    for _ in range(max_tries):
        try:
            mdl = PliableLasso(cv=cv_used, verbose=False, eps=1e-4, normalize=True,
                               max_interaction_terms=int(mit), alpha=alpha)
            mdl.fit(Xtr, Xtr[:, z_idx], ytr)
            return mdl, cv_used, None
        except Exception as e:
            err_msg = str(e)
            err = err_msg

            empty_path_like = ("argmin of an empty sequence" in err_msg) or ("empty" in err_msg.lower())
            next_cv = min(cv_used + CONFIG["CV_STEP_UP"], CONFIG["MAX_CV_FRACTION"])

            if (next_cv <= cv_used + 1e-12 or next_cv >= CONFIG["MAX_CV_FRACTION"] - 1e-12) and empty_path_like and not tried_smaller_mit:
                mit = max(3, int(np.ceil(mit / 2)))
                tried_smaller_mit = True
            else:
                cv_used = next_cv

            if cv_used >= CONFIG["MAX_CV_FRACTION"] - 1e-12 and tried_smaller_mit and empty_path_like:
                break

    return None, cv_used, err

def inner_cv_grid(X: np.ndarray, y: np.ndarray, z_idx: np.ndarray,
                  alpha_list: Sequence[float], cv_list: Sequence[float], mit_list: Sequence[int],
                  metric: str):
    rows = []
    best_score, best_alpha, best_cv, best_mit = -np.inf, None, None, None
    skf = _kfold_with_min_class(y, CONFIG["inner_folds"], RANDOM_STATE_INNER)

    for alpha in alpha_list:
        for cv in cv_list:
            for mit in mit_list:
                scores, tried_cvs, fails = [], [], 0
                for tr, va in skf.split(X, y):
                    mdl, cv_used, err = _fit_plasso_safe(X[tr], y[tr], z_idx, cv_init=cv, mit=mit, alpha=alpha)
                    if mdl is not None:
                        mdl.alpha = alpha
                        mdl.cv = cv_used
                        mdl.max_interaction_terms = mit
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
                    "max_interaction_terms": mit,
                    f"mean_{metric}": m,
                    f"std_{metric}": s,
                    "fold_failures": int(fails),
                })
                if np.isfinite(m) and (m > best_score or (m == best_score and (best_mit is None or mit < best_mit))):
                    best_score, best_alpha, best_cv, best_mit = m, alpha, cv, mit

    leaderboard = pd.DataFrame(rows).sort_values(by=f"mean_{metric}", ascending=False).reset_index(drop=True)
    if best_alpha is None or best_cv is None or best_mit is None:
        raise RuntimeError("All (alpha,cv,mit) combos failed to fit for this Z candidate.")
    return best_alpha, best_cv, best_mit, best_score, leaderboard

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
    mit_list: Sequence[int],
    metric: str
):
    cands = _build_z_candidates_categories(
        X, y, k=k, max_abs_corr=max_abs_corr,
        categories=categories, n_per_category=n_per_category,
        jitter=jitter, seed=seed
    )
    rows = []
    best = (None, None, None, None, -np.inf, None, None)
    for cid, (cat, z_idx) in enumerate(cands, 1):
        try:
            alpha_star, cv_star, mit_star, score_star, _ = inner_cv_grid(
                X, y, z_idx, alpha_list=alpha_list, cv_list=cv_list, mit_list=mit_list, metric=metric
            )
        except Exception as e:
            rows.append({"candidate_id": cid, "category": cat, "z_size": int(len(z_idx)),
                         "best_score": np.nan, "error": str(e),
                         "z_names": ";".join([x_cols[int(j)] for j in z_idx])})
            continue
        rows.append({"candidate_id": cid, "category": cat, "z_size": int(len(z_idx)),
                     "best_score": float(score_star),
                     "best_alpha": float(alpha_star), "best_cv": float(cv_star), "best_mit": int(mit_star),
                     "z_names": ";".join([x_cols[int(j)] for j in z_idx])})
        if score_star > best[4]:
            best = (z_idx, alpha_star, cv_star, mit_star, score_star, cid, cat)

    z_idx_best, alpha_best, cv_best, mit_best, score_best, cid_best, cat_best = best
    z_df = pd.DataFrame(rows).sort_values(by=["best_score"], ascending=False).reset_index(drop=True)
    if z_idx_best is None:
        raise RuntimeError("All Z candidates failed to fit. Consider reducing z_k or loosening z_max_abs_corr.")
    return z_idx_best, alpha_best, cv_best, mit_best, score_best, cid_best, cat_best, z_df

def outer_cv_eval_with_zsearch(X: np.ndarray, y: np.ndarray, x_cols: List[str]):
    skf = _kfold_with_min_class(y, CONFIG["outer_folds"], RANDOM_STATE_OUTER)
    oof = np.full_like(y, fill_value=np.nan, dtype=float)
    fold_rows = []
    rng_base = np.random.default_rng(CONFIG["z_random_state"])

    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        pos_tr, neg_tr = int(y[tr].sum()), int((y[tr]==0).sum())
        pos_te, neg_te = int(y[te].sum()), int((y[te]==0).sum())
        print(f"[fold {fold}] train pos={pos_tr} neg={neg_tr} | test pos={pos_te} neg={neg_te}")

        seed_fold = int(rng_base.integers(0, 1_000_000))
        try:
            z_idx, best_alpha, best_cv, best_mit, zscore, cid, cat, _ = choose_best_z_via_inner_cv_categories(
                X[tr], y[tr], x_cols,
                k=int(CONFIG["z_k"]),
                max_abs_corr=float(CONFIG["z_max_abs_corr"]),
                categories=CONFIG["z_categories"],
                n_per_category=int(CONFIG["z_n_per_category"]),
                jitter=float(CONFIG["z_jitter"]),
                seed=seed_fold,
                alpha_list=CONFIG["alpha_grid"],
                cv_list=CONFIG["cv_grid"],
                mit_list=CONFIG["mit_grid"],
                metric=CONFIG["primary_metric"]
            )
        except Exception as e_primary:
            print(f"[WARN] Fold {fold}: all Z candidates failed ({e_primary}). Trying fallback...")
            try:
                z_idx, best_alpha, best_cv, best_mit, zscore, cid, cat, _ = choose_best_z_via_inner_cv_categories(
                    X[tr], y[tr], x_cols,
                    k=int(CONFIG["z_k"]),
                    max_abs_corr=float(CONFIG["z_max_abs_corr_relaxed"]),
                    categories=["high_variance"],
                    n_per_category=max(2, int(CONFIG["z_n_per_category"])),
                    jitter=float(CONFIG["z_jitter"]),
                    seed=seed_fold + 1,
                    alpha_list=CONFIG["alpha_grid"],
                    cv_list=CONFIG["cv_grid_fallback"],
                    mit_list=CONFIG["mit_grid_fallback"],
                    metric=CONFIG["primary_metric"]
                )
            except Exception as e_fb:
                print(f"[WARN] Fold {fold}: fallback also failed ({e_fb}). Using last-resort high-variance Z.")
                vari = np.var(X[tr], axis=0, ddof=1)
                order = np.argsort(-vari)
                z_idx = _topk_from_order(order, int(CONFIG["z_k"]))
                best_alpha = CONFIG["alpha_grid"][0]
                best_cv = CONFIG["cv_grid_fallback"][-1]
                best_mit = CONFIG["mit_grid_fallback"][0]
                zscore = float("nan"); cid = -1; cat = "last_resort_high_variance"

        mdl, cv_used, err = _fit_plasso_safe(X[tr], y[tr], z_idx, cv_init=best_cv, mit=best_mit, alpha=best_alpha)
        if mdl is None:
            fold_rows.append({
                "fold": fold, "error": err,
                "roc_auc": np.nan, "average_precision": np.nan,
                "accuracy_t": np.nan, "f1_t": np.nan,
                "best_alpha": float(best_alpha), "best_cv": float(best_cv), "best_mit": int(best_mit),
                "z_candidate_id": int(cid), "z_category": cat,
                "z_names": ";".join([x_cols[int(j)] for j in z_idx]),
                "n_z": int(len(z_idx)),
                "cv_used_effective": np.nan,
            })
            continue

        scores_te = np.asarray(mdl.predict(X[te], X[te][:, z_idx])).ravel()
        scores_tr = np.asarray(mdl.predict(X[tr], X[tr][:, z_idx])).ravel()

        if CONFIG["threshold_strategy"] == "maximize_f1":
            t = _best_threshold_f1(y[tr], scores_tr)
        else:
            t = float(CONFIG["fixed_threshold"])

        yhat_te = (scores_te > t).astype(int)
        oof[te] = scores_te

        theta = get_theta(mdl)
        nnz_theta = int(np.sum(np.abs(theta) > 1e-12)) if theta is not None else 0

        def _safe(fn, *args):
            try: return fn(*args)
            except Exception: return np.nan

        fold_rows.append({
            "fold": fold,
            "roc_auc": _safe(roc_auc_score, y[te], scores_te),
            "average_precision": _safe(average_precision_score, y[te], scores_te),
            "accuracy_t": _safe(accuracy_score, y[te], yhat_te),
            "f1_t": _safe(f1_score, y[te], yhat_te),
            "threshold_used": float(t),
            "best_alpha": float(best_alpha),
            "best_cv": float(best_cv),
            "best_mit": int(best_mit),
            "z_candidate_id": int(cid),
            "z_category": cat,
            "z_inner_best_score": float(zscore) if np.isfinite(zscore) else np.nan,
            "z_names": ";".join([x_cols[int(j)] for j in z_idx]),
            "n_z": int(len(z_idx)),
            "cv_used_effective": float(cv_used),
            "nnz_theta": int(nnz_theta),
            "cap_hit_likely": int(nnz_theta >= int(best_mit)),
        })

    perf_df = pd.DataFrame(fold_rows)
    summary = {
        "roc_auc_mean":    _mean_no_warn(perf_df["roc_auc"].to_numpy()) if len(perf_df) else np.nan,
        "pr_auc_mean":     _mean_no_warn(perf_df["average_precision"].to_numpy()) if len(perf_df) else np.nan,
        "accuracy_t_mean": _mean_no_warn(perf_df["accuracy_t"].to_numpy()) if len(perf_df) else np.nan,
        "f1_t_mean":       _mean_no_warn(perf_df["f1_t"].to_numpy()) if len(perf_df) else np.nan,
        "threshold_mean":  _mean_no_warn(perf_df["threshold_used"].to_numpy()) if len(perf_df) else np.nan,
    }
    return oof, perf_df, summary

def fit_one(dataset_tag: str, path: str):
    Xdf, y, x_cols = read_tgs_csv(path)
    X = Xdf.to_numpy(dtype=float)
    p = X.shape[1]

    oof_scores, folds_df, summary = outer_cv_eval_with_zsearch(X, y, x_cols)

    try:
        z_idx_full, best_alpha_full, best_cv_full, best_mit_full, score_full, cid_full, cat_full, zcand_df_full = choose_best_z_via_inner_cv_categories(
            X, y, x_cols,
            k=int(CONFIG["z_k"]),
            max_abs_corr=float(CONFIG["z_max_abs_corr"]),
            categories=CONFIG["z_categories"],
            n_per_category=int(CONFIG["z_n_per_category"]),
            jitter=float(CONFIG["z_jitter"]),
            seed=int(CONFIG["z_random_state"]),
            alpha_list=CONFIG["alpha_grid"],
            cv_list=CONFIG["cv_grid"],
            mit_list=CONFIG["mit_grid"],
            metric=CONFIG["primary_metric"]
        )
    except Exception as e_full:
        print(f"[WARN] Full-data Z search failed ({e_full}). Trying fallback...")
        z_idx_full, best_alpha_full, best_cv_full, best_mit_full, score_full, cid_full, cat_full, zcand_df_full = choose_best_z_via_inner_cv_categories(
            X, y, x_cols,
            k=int(CONFIG["z_k"]),
            max_abs_corr=float(CONFIG["z_max_abs_corr_relaxed"]),
            categories=["high_variance"],
            n_per_category=max(2, int(CONFIG["z_n_per_category"])),
            jitter=float(CONFIG["z_jitter"]),
            seed=int(CONFIG["z_random_state"]) + 7,
            alpha_list=CONFIG["alpha_grid"],
            cv_list=CONFIG["cv_grid_fallback"],
            mit_list=CONFIG["mit_grid_fallback"],
            metric=CONFIG["primary_metric"]
        )

    mdl, cv_used, err = _fit_plasso_safe(X, y, z_idx_full, cv_init=best_cv_full, mit=best_mit_full, alpha=best_alpha_full)

    if mdl is None:
        print(f"[WARN] Final fit failed with ({best_cv_full}, MIT={best_mit_full}, alpha={best_alpha_full}): {err}")
        try:
            best_alpha_fb, cv_fb, mit_fb, _, _ = inner_cv_grid(
                X, y, z_idx_full,
                alpha_list=CONFIG["alpha_grid"],
                cv_list=CONFIG["cv_grid_fallback"],
                mit_list=CONFIG["mit_grid_fallback"],
                metric=CONFIG["primary_metric"]
            )
            print(f"[INFO] Retrying final fit with fallback hyperparams: alpha={best_alpha_fb}, cv={cv_fb}, MIT={mit_fb}")
            mdl, cv_used, err = _fit_plasso_safe(X, y, z_idx_full, cv_init=cv_fb, mit=mit_fb, alpha=best_alpha_fb)
            if mdl is not None:
                best_alpha_full, best_cv_full, best_mit_full = best_alpha_fb, cv_fb, mit_fb
        except Exception as e_fb:
            err = f"{err} | fallback inner-CV failed: {e_fb}"

    if mdl is None:
        print("[WARN] Final fit still failing; trying relaxed high-variance Z for final model only.")
        vari = np.var(X, axis=0, ddof=1)
        order = np.argsort(-vari)
        z_idx_relaxed = _topk_from_order(order, int(CONFIG["z_k"]))
        best_alpha_fb = CONFIG["alpha_grid"][0]
        cv_fb = CONFIG["cv_grid_fallback"][-1]
        mit_fb = CONFIG["mit_grid_fallback"][0]
        mdl, cv_used, err2 = _fit_plasso_safe(X, y, z_idx_relaxed, cv_init=cv_fb, mit=mit_fb, alpha=best_alpha_fb)
        if mdl is None:
            raise RuntimeError(f"Final fit failed for dataset {dataset_tag}: {err} | relaxed-Z failed: {err2}")
        else:
            z_idx_full = z_idx_relaxed
            best_alpha_full, best_cv_full, best_mit_full = best_alpha_fb, cv_fb, mit_fb
            cat_full = "final_relaxed_high_variance"

    beta = get_beta(mdl); theta = get_theta(mdl); intercept = get_intercept(mdl)

    stem = stem_for(dataset_tag)
    out_dir = os.path.dirname(os.path.abspath(path)) or "."

    folds_df.to_csv(os.path.join(out_dir, f"cv_folds_{stem}.csv"), index=False)
    if CONFIG["SAVE_PRED"]:
        pd.DataFrame({"oof_score": oof_scores, "y": y}).to_csv(
            os.path.join(out_dir, f"yhat_oof_{stem}.csv"), index=False
        )

    zcand_df_full.to_csv(os.path.join(out_dir, f"zsearch_leaderboard_{stem}.csv"), index=False)

    _, _, _, _, chosen_leaderboard = inner_cv_grid(
        X, y, z_idx_full,
        alpha_list=CONFIG["alpha_grid"],
        cv_list=CONFIG["cv_grid"],
        mit_list=CONFIG["mit_grid"],
        metric=CONFIG["primary_metric"]
    )
    chosen_leaderboard.to_csv(os.path.join(out_dir, f"leaderboard_{stem}.csv"), index=False)

    if CONFIG["SAVE_COEF_CSV"]:
        if beta is not None:
            beta = np.asarray(beta).ravel()
            pd.DataFrame(
                [{"j": int(j), "name": x_cols[j], "coef": float(b)} for j, b in enumerate(beta) if abs(b) > 1e-12]
            ).to_csv(os.path.join(out_dir, f"beta_{stem}.csv"), index=False)
        if theta is not None:
            rows = []
            p_mod = theta.shape[1]
            for i in range(p):
                for jj in range(p_mod):
                    val = float(theta[i, jj])
                    if abs(val) > 1e-12:
                        j = int(z_idx_full[jj])
                        rows.append({"i": int(i), "i_name": x_cols[i],
                                     "j": j, "j_name": x_cols[j], "coef": val})
            if rows:
                pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"theta_{stem}.csv"), index=False)

    t0 = time(); _ = np.asarray(mdl.predict(X, X[:, z_idx_full])).ravel(); t1 = time()

    beta_rows = []
    if beta is not None:
        for j, b in enumerate(np.asarray(beta).ravel()):
            if abs(b) > 1e-12:
                beta_rows.append({"j": int(j), "name": x_cols[j], "coef": float(b)})

    theta_rows = []
    if theta is not None:
        p_mod = theta.shape[1]
        for i in range(p):
            for jj in range(p_mod):
                val = float(theta[i, jj])
                if abs(val) > 1e-12:
                    j = int(z_idx_full[jj])
                    theta_rows.append({"i": int(i), "i_name": x_cols[i],
                                       "j": j,      "j_name": x_cols[j],
                                       "coef": val})

    # ==== Save selected mains/interactions to results/tgs_results ====
    results_dir = r"C:\Users\enthe\Desktop\Thesis\results\tgs_results"
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"selected_2_{stem}.json")

    selected_mains = sorted([x_cols[j] for j, b in enumerate(np.asarray(beta).ravel()) if abs(b) > 1e-12]) if beta is not None else []
    selected_interactions = [
        {"main": x_cols[i], "modifier": x_cols[int(z_idx_full[jj])], "coef": float(theta[i, jj])}
        for i in range(p) for jj in range(theta.shape[1])
        if theta is not None and abs(theta[i, jj]) > 1e-12
    ]

    y_pred = mdl.predict(X, X[:, z_idx_full]).ravel()
    y_true = y.ravel()
    y_pred_label = (y_pred > 0.5).astype(int)
    mse = np.mean((y_true - y_pred) ** 2)
    accuracy = accuracy_score(y_true, y_pred_label)

    results_dict = {
        "dataset": os.path.basename(path),
        "selected_hyperparams": {
            "alpha": float(best_alpha_full),
            "cv_requested": float(best_cv_full),
            "max_interaction_terms": int(best_mit_full)
        },
        "chosen_z_category": cat_full,
        "chosen_z_names": [x_cols[int(k)] for k in z_idx_full],
        "selected_mains": selected_mains,
        "selected_interactions": selected_interactions,
        "intercept": float(intercept),
        "fit_time_seconds": round(t1 - t0, 3),
        "mse": mse,
        "accuracy": accuracy,
        "outer_cv_metrics": summary
    }
    with open(results_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"[OK] Saved selected mains/interactions to {results_path}")

    print(f"[OK] {dataset_tag}: PR-AUC={summary['pr_auc_mean']:.3f}, ROC-AUC={summary['roc_auc_mean']:.3f} | saved with stem 'tgs_{dataset_tag}'")
    return summary

def main():
    # Uncomment the dataset you want to run
    # path = CONFIG["BALANCED_PATH"]
    path = CONFIG["IMBALANCED_PATH"]
    tag = "2imbalanced"  # or "balanced" if using balanced

    #path = CONFIG["BALANCED_PATH"]
    #tag = "balanced"  # or "balanced" if using balanced



    if not os.path.exists(path):
        print(f"[WARN] {path} not found; exiting.")
        return

    summary = fit_one(tag, path)
    with open("tgs_compare.json", "w") as f:
        json.dump({tag: summary}, f, indent=2)
if __name__ == "__main__":
    main()