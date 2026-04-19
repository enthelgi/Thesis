#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
- Iterates over all combos:
    n = base_n * n_multiplier
    p = base_p * p_multiplier
    prev ∈ prevalences
    rho ∈ rhos
- For each dataset "<DATA_DIR>/simulated_dataset_{n}_{prev}_{p}_{rho}.csv":
    * loads truth at "truth_<stem>.json"
    * fits Pliable Lasso (grid on cv × max_interaction_terms)
    * prints F1s + accuracy at cutoff
    * saves:
        chosen_<stem>.json
        best_model_<stem>.json
        [optional] yhat_<stem>.csv, beta_<stem>.csv, theta_<stem>.csv
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
# Configuration 
# ==========================

CONFIG = {
    # --- WHERE the CSVs live ---
    "DATA_DIR": r"C:/Users/enthe/Desktop/Thesis/data/simulated_data",  

    # --- Batch design (54 datasets) ---
    "base_n": 500,
    "base_p": 20,
    "n_multipliers": (0.5, 1.0, 2.0),
    "prevalences": (0.05, 0.20, 0.50),
    "p_multipliers": (1, 25),
    "rhos": (0.0, 0.5, 0.8),

    # If a file is missing, skip it (True) or raise (False)
    "SKIP_MISSING": True,

    # --- Model/grid & outputs ---
    "cv_grid": [0.05, 0.1, 0.2, 0.3],
    "mit_grid": [5, 10, 20, 50, 100],
    "threshold": 0.5,       # cutoff for hard class prediction
    "mod_k": 10,            # if no truth, select k low-corr columns for Z

    # Per-file outputs use prefix style automatically; these toggles just enable/disable extras
    "SAVE_PRED": False,           # also save yhat_<stem>.csv
    "SAVE_COEF_CSV": True,        # also save beta_<stem>.csv and theta_<stem>.csv
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

def make_stem(n: int, prev: float, p: int, rho: float) -> str:
    return f"simulated_dataset_{n}_{_fmt_f(prev)}_{p}_{_fmt_f(rho)}"

def dataset_paths_from_config(cfg: Dict) -> List[str]:
    stems = []
    for nm, prev, pm, rho in product(
        cfg["n_multipliers"], cfg["prevalences"], cfg["p_multipliers"], cfg["rhos"]
    ):
        n = int(round(cfg["base_n"] * nm))
        p = int(round(cfg["base_p"] * pm))
        stems.append(make_stem(n, prev, p, rho))
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
    truth = os.path.join(dirpath, f"truth_{stem}.json")  
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
    FN = len(true_set - pred_set)
    prec = TP / (TP + FP) if (TP + FP) else 0.0
    rec = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return f1, prec, rec, TP, FP, FN

# -------------------------
# Fit core
# -------------------------

def grid_search_best_model(
    X: np.ndarray,
    y: np.ndarray,
    mains_true: Set[int],
    inter_true: Set[frozenset],
    z_idx: Optional[Sequence[int]] = None,
    cv_list: Sequence[float] = (0.05, 0.1, 0.2, 0.3),
    mit_list: Sequence[int] = (5, 10, 20, 50, 100),
    verbose: bool = True,
):
    p = X.shape[1]
    z_idx = np.arange(p, dtype=int) if z_idx is None else np.asarray(z_idx, dtype=int)
    Z = X[:, z_idx]
    p_mod = Z.shape[1]

    best_f1, best_rec = -1.0, -1.0
    best_mit, best_cv = None, None
    best_model = None
    rows = []

    for cv in cv_list:
        for mit in mit_list:
            if verbose:
                print(f"\n=== Trying cv={cv}, max_interaction_terms={mit} ===")
            mdl = PliableLasso(cv=cv, verbose=False, eps=1e-4, normalize=True, max_interaction_terms=mit)
            t0 = time()
            mdl.fit(X, Z, y)
            fit_time = time() - t0
            if verbose:
                print(f"  fitted in {fit_time:.2f}s")

            theta = get_theta(mdl)
            pred_inter: Set[frozenset] = set()
            if theta is not None:
                for i in range(p):
                    for jj in range(p_mod):
                        if abs(theta[i, jj]) > 1e-8:
                            j = int(z_idx[jj])
                            pred_inter.add(frozenset((i, j)))

            f1_i, prec_i, rec_i, TP_i, FP_i, FN_i = f1_from_sets(inter_true, pred_inter)
            rows.append({
                "cv": cv, "max_interaction_terms": mit,
                "F1_inter": f1_i, "precision_inter": prec_i, "recall_inter": rec_i,
                "TP_inter": TP_i, "FP_inter": FP_i, "FN_inter": FN_i,
                "n_pred_interactions": len(pred_inter), "fit_time_s": fit_time
            })

            better = (
                (f1_i > best_f1) or
                (f1_i == best_f1 and rec_i > best_rec) or
                (f1_i == best_f1 and rec_i == best_rec and (best_mit is None or mit < best_mit))
            )
            if better:
                best_f1, best_rec = f1_i, rec_i
                best_mit, best_cv = mit, cv
                best_model = mdl

    if verbose:
        print(f'\n>>> Selected cv={best_cv}, max_interaction_terms={best_mit} '
              f'with Interaction F1={best_f1:.3f} (Rec={best_rec:.3f})\n')

    return best_model, pd.DataFrame(rows), best_cv, best_mit

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

        z_idx = derive_z_indices(truth, X, k_if_no_truth=int(CONFIG["mod_k"]))

        best_model, _leaderboard, best_cv, best_mit = grid_search_best_model(
            X, y, mains_true, inter_true,
            z_idx=z_idx,
            cv_list=CONFIG["cv_grid"], mit_list=CONFIG["mit_grid"], verbose=True
        )

        Z = X[:, z_idx]
        y_pred = predict_classes(best_model, X, Z=Z, threshold=float(CONFIG["threshold"]))

        n_total = int(y.shape[0])
        n_correct = int((y_pred == y).sum())
        accuracy = float(n_correct / n_total) if n_total else 0.0

        p = X.shape[1]
        mains_pred, inter_pred = extract_selected_sets(best_model, p, z_idx=z_idx)

        f1_m, prec_m, rec_m, TP_m, FP_m, FN_m = f1_from_sets(mains_true, mains_pred)
        f1_i, prec_i, rec_i, TP_i, FP_i, FN_i = f1_from_sets(inter_true, inter_pred)

        print("\n=== Performance on selection ===")
        print(f"[{stem}] Mains:        F1={f1_m:.3f} (P={prec_m:.3f}, R={rec_m:.3f}) | TP={TP_m} FP={FP_m} FN={FN_m}")
        print(f"[{stem}] Interactions: F1={f1_i:.3f} (P={prec_i:.3f}, R={rec_i:.3f}) | TP={TP_i} FP={FP_i} FN={FN_i}")
        print(f"[{stem}] Accuracy @ cutoff {CONFIG['threshold']:.3f}: {accuracy:.4f}  "
              f"({n_correct}/{n_total} correct)")

        # ---- Save selections & metrics (prefix style) ----
        out_json = os.path.join(out_dir, f"chosen_{stem}.json")
        chosen = {
            "selected_mains": sorted([int(i) for i in mains_pred]),
            "selected_interactions": sorted([sorted(list(s)) for s in inter_pred]),
            "selected_hyperparams": {
                "cv": float(best_cv),
                "max_interaction_terms": int(best_mit),
                "threshold": float(CONFIG["threshold"])
            },
            "metrics": {
                "accuracy_at_threshold": float(accuracy),
                "n_correct": int(n_correct),
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
            "hyperparams": {"cv": float(best_cv), "max_interaction_terms": int(best_mit)},
            "threshold": float(CONFIG["threshold"]),
            "intercept": float(intercept),
            "beta": beta_rows,
            "theta": theta_rows
        }
        model_json_path = os.path.join(out_dir, f"best_model_{stem}.json")
        with open(model_json_path, "w") as f:
            json.dump(model_bundle, f, indent=2)
        print(f"[{stem}] Saved best model parameters → {model_json_path}")

        if CONFIG["SAVE_COEF_CSV"]:
            if beta_rows:
                beta_csv = os.path.join(out_dir, f"beta_{stem}.csv")
                pd.DataFrame(beta_rows).to_csv(beta_csv, index=False)
                print(f"[{stem}] Saved nonzero beta → {beta_csv}")
            if theta_rows:
                theta_csv = os.path.join(out_dir, f"theta_{stem}.csv")
                pd.DataFrame(theta_rows).to_csv(theta_csv, index=False)
                print(f"[{stem}] Saved nonzero theta → {theta_csv}")

        if CONFIG["SAVE_PRED"]:
            pred_path = os.path.join(out_dir, f"yhat_{stem}.csv")
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
