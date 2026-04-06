#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fit_and_f1.py  — batch runner (simulated continuous datasets), JSON/CSV saves, prefix-style filenames

- Iterates over all combos:
    n = base_n * n_multiplier
    noise_sd ∈ noise_sds
    p = base_p * p_multiplier
    rho ∈ rhos
- For each dataset "<DATA_DIR>/simulated_dataset_{n}_{noise_sd}_{p}_{rho}.csv":
    * loads truth at "truth_<stem>.json"
    * derives Z
    * runs OUTER K-fold CV (fixed hyperparams) + permutation rounds for evaluation (printed)
    * fits Pliable Lasso once on FULL data with fixed hyperparams
    * prints final F1s + MSE
    * saves:
        chosen_continuous_<stem>.json
        best_model_continuous_<stem>.json
        [optional] yhat_continuous_<stem>.csv, beta_continuous_<stem>.csv, theta_continuous_<stem>.csv
- No CLI. Configure CONFIG below.
"""

import json
import os
import re
from time import time
from typing import Dict, Optional, Sequence, Set, Tuple, List
from itertools import product

import numpy as np
import pandas as pd
from plasso import PliableLasso

# ==========================
# Configuration (no CLI)
# ==========================

CONFIG = {
    # --- WHERE the CSVs live ---
    "DATA_DIR": r"C:/Users/enthe/Desktop/Thesis/data/simulated_data_continuous",
    "USE_FULL_X_AS_Z": False,
    "Z_EXCLUDE_ONE_NON_EFFECT": False,  # Set True to use this option

    # --- Batch design ---
    "base_n": 500,
    "base_p": 20,
    "n_multipliers": (0.5, 1.0, 2.0),
    "noise_sds": (0.5, 1.0, 2.0),
    "p_multipliers": (1, 25),
    "rhos": (0.0, 0.5, 0.8),

    # If a file is missing, skip it (True) or raise (False)
    "SKIP_MISSING": True,

    # --- Model & outputs ---
    # IMPORTANT: cv values MUST be < 1.0 (interpreted by plasso as test_size fraction).
    "cv_grid": [0.2],       # fixed inner test_size fraction used inside PliableLasso
    "mit_grid": [70],      # fixed max_interaction_terms
    "outer_k": 5,           # outer K-fold for evaluation
    "n_permutations": 2,    # number of permutation rounds for outer CV evaluation
    "threshold": 0.5,       # cutoff for hard class prediction
    "mod_k": 10,            # if no truth, select k low-corr columns for Z

    # Per-file outputs use prefix style automatically; these toggles just enable/disable extras
    "SAVE_PRED": False,           # also save yhat_continuous_<stem>.csv
    "SAVE_COEF_CSV": False,       # also save beta_continuous_<stem>.csv and theta_continuous_<stem>.csv
}

# -------------------------
# Helpers: naming & I/O
# -------------------------

def _fmt_f(x: float) -> str:
    """Format float for filenames: 1 decimal if multiple of 0.1, else 2 decimals."""
    x = float(x)
    if abs(x - round(x, 1)) < 1e-12:
        return f"{x:.1f}"
    return f"{x:.2f}"

def make_stem(n: int, noise_sd: float, p: int, rho: float) -> str:
    return f"simulated_dataset_{n}_{_fmt_f(noise_sd)}_{p}_{_fmt_f(rho)}"

def dataset_paths_from_config(cfg: Dict) -> List[str]:
    stems = []
    for nm, noise_sd, pm, rho in product(
        cfg["n_multipliers"], cfg["noise_sds"], cfg["p_multipliers"], cfg["rhos"]
    ):
        n = int(round(cfg["base_n"] * nm))
        p = int(round(cfg["base_p"] * pm))
        stems.append(make_stem(n, noise_sd, p, rho))
    return [os.path.join(cfg["DATA_DIR"], f"{stem}.csv") for stem in stems]

def parse_sim_filename(path: str) -> Tuple[str, str, str]:
    """
    Return (csv_path, truth_json_path, out_stem). Truth = prefix "truth_<stem>.json".
    out_stem = "<dir>/<stem>" used to derive all outputs.
    """
    csv_path = path
    base = os.path.basename(path)
    m = re.match(r"(simulated_dataset_\d+_[0-9.]+_\d+_[0-9.]+)\.csv$", base)
    if not m:
        raise ValueError(f"Bad data filename: {base}")
    stem = m.group(1)
    dirpath = os.path.dirname(path)
    truth = os.path.join(dirpath, f"truth_{stem}.json")  # prefix
    out_stem = os.path.join(dirpath, stem)
    return csv_path, truth, out_stem

def read_sim_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    import re as _re
    x_cols = [c for c in df.columns if _re.fullmatch(r'[xX]\d+', c)]
    y_col = next((c for c in df.columns if c.lower() == 'y'), None)

    if x_cols and y_col:
        def x_index(name: str) -> int:
            return int(_re.findall(r'\d+', name)[0])
        x_cols_sorted = sorted(x_cols, key=x_index)
        X = df[x_cols_sorted].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
        y = pd.to_numeric(df[y_col], errors='coerce').to_numpy(dtype=float)
        if np.isnan(X).any() or np.isnan(y).any():
            raise ValueError("Non-numeric values found in X/y.")
        return X, y

    df = pd.read_csv(path, header=None)
    if df.shape[1] < 3:
        raise ValueError("CSV must have at least 3 columns (X..., y, p_i).")
    X = df.iloc[:, :-2].to_numpy(dtype=float)
    y = df.iloc[:, -2].to_numpy(dtype=float)
    return X, y

def read_truth_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)

# -------------------------
# Model helpers
# -------------------------

def select_modifiers_by_low_corr(X: np.ndarray, k: int) -> np.ndarray:
    p = X.shape[1]
    k = max(1, min(k, p))
    Xc = X - X.mean(0, keepdims=True)
    C = np.corrcoef(Xc, rowvar=False)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(C, 0.0)
    avg_abs = np.mean(np.abs(C), axis=0)
    return np.argsort(avg_abs)[:k].astype(int)

def z_indices_from_truth(truth: Dict, p: int) -> np.ndarray:
    pairs = truth.get("interaction_pairs", [])
    z = sorted({i for pair in pairs for i in pair}) if pairs else list(range(p))
    return np.asarray(z, dtype=int)

def derive_z_indices(truth: Dict, X: np.ndarray, k_if_no_truth: int) -> np.ndarray:
    pairs = truth.get("interaction_pairs", []) if truth else []
    if pairs:
        return z_indices_from_truth(truth, X.shape[1])
    return select_modifiers_by_low_corr(X, k=k_if_no_truth)

def find_non_effect_indices(truth: Dict, p: int) -> list:
    mains = set(truth.get("main_idx", []))
    inter = set()
    for pair in truth.get("interaction_pairs", []):
        inter.update(pair)
    all_effects = mains | inter
    return [idx for idx in range(p) if idx not in all_effects]

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

def truth_sets(truth: Dict) -> Tuple[Set[int], Set[frozenset]]:
    mains = set(truth.get("main_idx", []))
    pairs = {frozenset(pair) for pair in truth.get("interaction_pairs", [])}
    return mains, pairs

def f1_from_sets(true_set: Set, pred_set: Set) -> Tuple[float, float, float, int, int, int]:
    TP = len(true_set & pred_set)
    FP = len(pred_set - true_set)
    FN = len(true_set - pred_set)  # <-- fixed: FN should be true minus predicted
    prec = TP / (TP + FP) if (TP + FP) else 0.0
    rec = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return f1, prec, rec, TP, FP, FN

def kfold_indices(n_samples: int,
                  n_splits: int,
                  shuffle: bool = True,
                  random_state: Optional[int] = 42):
    """
    Simple K-fold splitter yielding (train_idx, val_idx) pairs.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2 for K-fold CV.")

    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

    folds = np.array_split(indices, n_splits)
    for k in range(n_splits):
        val_idx = folds[k]
        train_idx = np.concatenate([folds[i] for i in range(n_splits) if i != k])
        yield train_idx, val_idx

# -------------------------
# Fit core (no hyperparam tuning)
# -------------------------

def extract_selected_sets(model, p: int, z_idx: Optional[Sequence[int]] = None) -> Tuple[Set[int], Set[frozenset]]:
    beta = get_beta(model)
    theta = get_theta(model)
    mains_pred: Set[int] = set()
    inter_pred: Set[frozenset] = set()

    if beta is not None:
        for j in range(p):
            if abs(beta[j]) > 1e-8:
                mains_pred.add(j)

    if theta is not None:
        z_idx = np.arange(p, dtype=int) if z_idx is None else np.asarray(z_idx, dtype=int)
        p_mod = theta.shape[1]
        for i in range(p):
            for jj in range(p_mod):
                if abs(theta[i, jj]) > 1e-8:
                    j = int(z_idx[jj])
                    inter_pred.add(frozenset((i, j)))

    return mains_pred, inter_pred

def predict_classes(model, X: np.ndarray, Z: Optional[np.ndarray] = None, threshold: float = 0.5) -> np.ndarray:
    if Z is None:
        Z = X
    if hasattr(model, "predict_proba"):
        yhat = model.predict_proba(X, Z)
    else:
        yhat = model.predict(X, Z)
    yhat = np.asarray(yhat).reshape(-1)
    return (yhat > threshold).astype(int)

def fit_pliable_lasso(
    X: np.ndarray,
    y: np.ndarray,
    z_idx: Optional[Sequence[int]],
    cv: float,
    max_interaction_terms: int,
    verbose: bool = True,
):
    """
    Fit PliableLasso ONCE on the full dataset with fixed hyperparameters.
    """
    if cv >= 1.0:
        raise ValueError(
            f"cv must be < 1.0 (test_size fraction) when doing manual outer CV, got {cv}."
        )

    p = X.shape[1]
    z_idx = np.arange(p, dtype=int) if z_idx is None else np.asarray(z_idx, dtype=int)
    Z = X[:, z_idx]

    model = PliableLasso(
        cv=cv,
        verbose=False,
        eps=1e-4,
        normalize=True,
        max_interaction_terms=max_interaction_terms,
        n_lam=100
    )
    t0 = time()
    model.fit(X, Z, y)
    fit_time = time() - t0

    if verbose:
        print(f"Fitted final PliableLasso on full data in {fit_time:.2f}s "
              f"(cv={cv}, max_interaction_terms={max_interaction_terms}).")

    return model

def _outer_cv_single_run(
    X: np.ndarray,
    y: np.ndarray,
    mains_true: Set[int],
    inter_true: Set[frozenset],
    z_idx: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    cv: float,
    max_interaction_terms: int,
    threshold: float,
    verbose: bool,
    tag: str,
    run_label: str,
):
    """
    One outer CV run (either on true y or on a permuted y).
    Returns dict of mean metrics across folds.
    """
    n_samples, p = X.shape
    Z = X[:, z_idx]

    f1_m_list, prec_m_list, rec_m_list = [], [], []
    f1_i_list, prec_i_list, rec_i_list = [], [], []
    mse_list = []
    n_pred_inter_list = []
    fit_time_list = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_tr, X_val = X[train_idx], X[val_idx]
        Z_tr, Z_val = Z[train_idx], Z[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        mdl = PliableLasso(
            cv=cv,
            verbose=False,
            eps=1e-4,
            normalize=True,
            max_interaction_terms=max_interaction_terms,
        )
        t0 = time()
        mdl.fit(X_tr, Z_tr, y_tr)
        fit_time = time() - t0
        fit_time_list.append(fit_time)

        mains_pred, inter_pred = extract_selected_sets(mdl, p, z_idx=z_idx)

        f1_m, prec_m, rec_m, TP_m, FP_m, FN_m = f1_from_sets(mains_true, mains_pred)
        f1_i, prec_i, rec_i, TP_i, FP_i, FN_i = f1_from_sets(inter_true, inter_pred)

        f1_m_list.append(f1_m)
        prec_m_list.append(prec_m)
        rec_m_list.append(rec_m)

        f1_i_list.append(f1_i)
        prec_i_list.append(prec_i)
        rec_i_list.append(rec_i)
        n_pred_inter_list.append(len(inter_pred))

        # Predictive metric on validation set
        y_val_pred = predict_classes(mdl, X_val, Z=Z_val, threshold=threshold)
        mse = float(np.mean((y_val_pred - y_val) ** 2))
        mse_list.append(mse)

        if verbose:
            print(
                f"{tag}{run_label} Fold {fold_idx+1}/{len(folds)}: "
                f"F1_mains={f1_m:.3f} (P={prec_m:.3f}, R={rec_m:.3f}) | "
                f"F1_inter={f1_i:.3f} (P={prec_i:.3f}, R={rec_i:.3f}) | "
                f"n_pred_inter={len(inter_pred)} | "
                f"MSE_val={mse:.4f} | fit_time={fit_time:.2f}s"
            )

    metrics = {
        "mean_f1_mains": float(np.mean(f1_m_list)) if f1_m_list else 0.0,
        "mean_precision_mains": float(np.mean(prec_m_list)) if prec_m_list else 0.0,
        "mean_recall_mains": float(np.mean(rec_m_list)) if rec_m_list else 0.0,
        "mean_f1_inter": float(np.mean(f1_i_list)) if f1_i_list else 0.0,
        "mean_precision_inter": float(np.mean(prec_i_list)) if prec_i_list else 0.0,
        "mean_recall_inter": float(np.mean(rec_i_list)) if rec_i_list else 0.0,
        "mean_n_pred_inter": float(np.mean(n_pred_inter_list)) if n_pred_inter_list else 0.0,
        "mean_mse_val": float(np.mean(mse_list)) if mse_list else 0.0,
        "mean_fit_time": float(np.mean(fit_time_list)) if fit_time_list else 0.0,
    }

    if verbose:
        print(
            f"{tag}{run_label} SUMMARY over {len(folds)} folds: "
            f"F1_mains={metrics['mean_f1_mains']:.3f} "
            f"(P={metrics['mean_precision_mains']:.3f}, R={metrics['mean_recall_mains']:.3f}) | "
            f"F1_inter={metrics['mean_f1_inter']:.3f} "
            f"(P={metrics['mean_precision_inter']:.3f}, R={metrics['mean_recall_inter']:.3f}) | "
            f"mean_n_pred_inter={metrics['mean_n_pred_inter']:.1f} | "
            f"mean_MSE_val={metrics['mean_mse_val']:.4f} | "
            f"mean_fit_time={metrics['mean_fit_time']:.2f}s"
        )

    return metrics

def run_outer_cv_with_permutations(
    X: np.ndarray,
    y: np.ndarray,
    mains_true: Set[int],
    inter_true: Set[frozenset],
    z_idx: Optional[Sequence[int]] = None,
    cv: float = 0.2,
    max_interaction_terms: int = 100,
    outer_k: int = 5,
    n_permutations: int = 2,
    threshold: float = 0.5,
    random_state: int = 42,
    verbose: bool = True,
    stem: Optional[str] = None,
):
    """
    Run OUTER K-fold CV with fixed hyperparameters, then run permutation rounds
    (on shuffled y) using the SAME outer folds. Metrics are printed for inspection.
    JSON formats remain unchanged (this function only evaluates).
    """
    n_samples, p = X.shape
    if outer_k < 2:
        raise ValueError("outer_k must be >= 2.")
    if outer_k > n_samples:
        raise ValueError(f"outer_k={outer_k} > n_samples={n_samples}.")

    if cv >= 1.0:
        raise ValueError(
            f"cv must be < 1.0 (test_size fraction) when doing manual outer CV, got {cv}."
        )

    z_idx = np.arange(p, dtype=int) if z_idx is None else np.asarray(z_idx, dtype=int)
    tag = f"[{stem}] " if stem else ""

    if verbose:
        print(
            f"{tag}Running OUTER {outer_k}-fold CV with PliableLasso "
            f"(cv={cv}, max_interaction_terms={max_interaction_terms})"
        )

    # Precompute folds once and reuse for permutations
    folds = list(kfold_indices(n_samples, outer_k, shuffle=True, random_state=random_state))

    # Original (unpermuted) outer CV
    _outer_cv_single_run(
        X, y, mains_true, inter_true,
        z_idx=z_idx,
        folds=folds,
        cv=cv,
        max_interaction_terms=max_interaction_terms,
        threshold=threshold,
        verbose=verbose,
        tag=tag,
        run_label="Original",
    )

    # Permutation rounds
    rng = np.random.default_rng(random_state + 12345)
    for perm_id in range(n_permutations):
        y_perm = rng.permutation(y)
        if verbose:
            print(f"{tag}Permutation round {perm_id+1}/{n_permutations} (shuffled y)")

        _outer_cv_single_run(
            X, y_perm, mains_true, inter_true,
            z_idx=z_idx,
            folds=folds,
            cv=cv,
            max_interaction_terms=max_interaction_terms,
            threshold=threshold,
            verbose=verbose,
            tag=tag,
            run_label=f"Perm{perm_id+1}",
        )

# -------------------------
# Per-dataset run
# -------------------------

def run_single_dataset(csv_path: str):
    try:
        csv_path, truth_path, out_stem = parse_sim_filename(csv_path)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(csv_path)
        if not os.path.exists(truth_path):
            raise FileNotFoundError(truth_path)

        out_dir = os.path.dirname(out_stem)
        stem = os.path.basename(out_stem)

        X, y = read_sim_csv(csv_path)
        truth = read_truth_json(truth_path)
        mains_true, inter_true = truth_sets(truth)

        # --- Derive Z indices ---
        if CONFIG.get("Z_EXCLUDE_ONE_NON_EFFECT", False):
            # Exclude all but N non-effect features from Z
            p = X.shape[1]
            non_effect_indices = find_non_effect_indices(truth, p)
            n_exclude = 15  # <-- set how many non-effect features to exclude from Z
            if len(non_effect_indices) >= n_exclude:
                exclude = non_effect_indices[:n_exclude]
                z_idx = np.array([i for i in range(p) if i not in exclude])
                print(f"[{stem}] Using all X except features {exclude} as Z (non-effect features excluded)")
            else:
                z_idx = np.arange(p)
                print(f"[{stem}] Not enough non-effect features; using full X as Z")
        elif CONFIG.get("USE_FULL_X_AS_Z", False):
            z_idx = np.arange(X.shape[1])
            print(f"[{stem}] Using full X as Z.")
        else:
            z_idx = derive_z_indices(truth, X, k_if_no_truth=int(CONFIG["mod_k"]))
            print(f"[{stem}] Using derived Z indices: {z_idx.tolist()}")

        # --- Fixed hyperparameters (no F1-based tuning) ---
        cv_val = float(CONFIG["cv_grid"][0])
        mit_val = int(CONFIG["mit_grid"][0])
        outer_k = int(CONFIG.get("outer_k", 5))
        n_perm = int(CONFIG.get("n_permutations", 2))
        threshold = float(CONFIG["threshold"])

        # --- Outer CV + permutation evaluation (printed only) ---
        run_outer_cv_with_permutations(
            X, y, mains_true, inter_true,
            z_idx=z_idx,
            cv=cv_val,
            max_interaction_terms=mit_val,
            outer_k=outer_k,
            n_permutations=n_perm,
            threshold=threshold,
            random_state=42,
            verbose=True,
            stem=stem,
        )

        # --- Fit final model on FULL data with fixed hyperparams ---
        best_model = fit_pliable_lasso(
            X, y,
            z_idx=z_idx,
            cv=cv_val,
            max_interaction_terms=mit_val,
            verbose=True,
        )

        # --- Evaluate final model on FULL data (for JSON metrics) ---
        Z = X[:, z_idx]
        y_pred = predict_classes(best_model, X, Z=Z, threshold=threshold)
        y_pred = np.asarray(y_pred).reshape(-1)
        n_total = int(y.shape[0])
        mse = float(np.mean((y_pred - y) ** 2))

        p = X.shape[1]
        mains_pred, inter_pred = extract_selected_sets(best_model, p, z_idx=z_idx)

        f1_m, prec_m, rec_m, TP_m, FP_m, FN_m = f1_from_sets(mains_true, mains_pred)
        f1_i, prec_i, rec_i, TP_i, FP_i, FN_i = f1_from_sets(inter_true, inter_pred)

        print("\n=== Performance on selection (FULL DATA, final model) ===")
        print(f"[{stem}] Mains:        F1={f1_m:.3f} (P={prec_m:.3f}, R={rec_m:.3f}) | TP={TP_m} FP={FP_m} FN={FN_m}")
        print(f"[{stem}] Interactions: F1={f1_i:.3f} (P={prec_i:.3f}, R={rec_i:.3f}) | TP={TP_i} FP={FP_i} FN={FN_i}")
        print(f"[{stem}] MSE: {mse:.4f}")

        # ---- Save selections & metrics (prefix style) ----
        out_json = os.path.join(out_dir, f"chosen_continuous_{stem}.json")
        chosen = {
            "selected_mains": sorted([int(i) for i in mains_pred]),
            "selected_interactions": sorted([sorted(list(s)) for s in inter_pred]),
            "selected_hyperparams": {
                "cv": float(cv_val),                    # JSON format unchanged
                "max_interaction_terms": int(mit_val),  # JSON format unchanged
                "threshold": float(threshold)
            },
            "metrics": {
                "mse": float(mse),
                "n_total": int(n_total),
                "f1_mains": float(f1_m),
                "precision_mains": float(prec_m),
                "recall_mains": float(rec_m),
                "f1_interactions": float(f1_i),
                "precision_interactions": float(prec_i),
                "recall_interactions": float(rec_i)
            }
        }
        with open(out_json, "w") as f:
            json.dump(chosen, f, indent=2)
        print(f"[{stem}] Saved selections & metrics → {out_json}")

        # ---- Save best model parameters (prefix style) ----
        beta = get_beta(best_model)
        theta = get_theta(best_model)
        intercept = get_intercept(best_model)

        beta_rows = []
        if beta is not None:
            for j, b in enumerate(beta):
                if abs(b) > 1e-12:
                    beta_rows.append({"j": int(j), "coef": float(b)})

        theta_rows = []
        if theta is not None:
            p_mod = theta.shape[1]
            for i in range(p):
                for jj in range(p_mod):
                    val = float(theta[i, jj])
                    if abs(val) > 1e-12:
                        j = int(z_idx[jj])
                        theta_rows.append({"i": int(i), "j": j, "coef": val})

        model_bundle = {
            "model_type": "pliable_lasso",
            "z_idx": [int(k) for k in np.asarray(z_idx, dtype=int)],
            "hyperparams": {
                "cv": float(cv_val),
                "max_interaction_terms": int(mit_val)
            },
            "threshold": float(threshold),
            "intercept": float(intercept),
            "beta": beta_rows,
            "theta": theta_rows
        }
        model_json_path = os.path.join(out_dir, f"best_model_continuous_{stem}.json")
        with open(model_json_path, "w") as f:
            json.dump(model_bundle, f, indent=2)
        print(f"[{stem}] Saved best model parameters → {model_json_path}")

        if CONFIG["SAVE_COEF_CSV"]:
            if beta_rows:
                beta_csv = os.path.join(out_dir, f"beta_continuous_{stem}.csv")
                pd.DataFrame(beta_rows).to_csv(beta_csv, index=False)
                print(f"[{stem}] Saved nonzero beta → {beta_csv}")
            if theta_rows:
                theta_csv = os.path.join(out_dir, f"theta_continuous_{stem}.csv")
                pd.DataFrame(theta_rows).to_csv(theta_csv, index=False)
                print(f"[{stem}] Saved nonzero theta → {theta_csv}")

        if CONFIG["SAVE_PRED"]:
            pred_path = os.path.join(out_dir, f"yhat_continuous_{stem}.csv")
            pd.DataFrame({"y_pred": y_pred}).to_csv(pred_path, index=False)
            print(f"[{stem}] Saved predicted classes → {pred_path}")

        return True, stem, None
    except Exception as e:
        return False, os.path.basename(os.path.splitext(csv_path)[0]), str(e)

# -------------------------
# Batch driver
# -------------------------

def main():
    csv_paths = dataset_paths_from_config(CONFIG)
    print(f"[INFO] Planned datasets: {len(csv_paths)}")

    successes, failures = [], []
    for path in csv_paths:
        stem = os.path.basename(path)[:-4]
        if not os.path.exists(path):
            msg = f"missing CSV {path}"
            if CONFIG["SKIP_MISSING"]:
                print(f"[SKIP] {msg}")
                failures.append((stem, msg))
                continue
            else:
                raise FileNotFoundError(msg)

        print(f"\n================= Running: {stem} =================")
        ok, s, err = run_single_dataset(path)
        if ok:
            successes.append(s)
        else:
            print(f"[ERROR] {s}: {err}")
            failures.append((s, err))

    print("\n=== Batch summary ===")
    print(f"Success: {len(successes)}  |  Failed/Skipped: {len(failures)}")
    if failures:
        for s, err in failures:
            print(f" - {s}: {err}")

if __name__ == "__main__":
    main()
