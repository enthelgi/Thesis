import json
import os
from time import time
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from plasso import PliableLasso
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, accuracy_score, average_precision_score, roc_auc_score

# ----------- CONFIGURATION -----------
PRIMARY_METRIC = "roc_auc"
ALPHAS     = [0.1, 0.3, 0.5, 0.7]
CVS        = [0.05, 0.1, 0.2, 0.3, 0.4]
N_OUTER, N_INNER, RANDOM_STATE, THR = 3, 2, 42, 1e-4

def _score_metric(y_true, y_pred, metric):
    if metric == "average_precision":
        return average_precision_score(y_true, y_pred)
    elif metric == "roc_auc":
        return roc_auc_score(y_true, y_pred)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

def _f1(T, P):
    tp = len(T & P)
    prec = tp / len(P) if P else 0.0
    rec = tp / len(T) if T else 0.0
    return (0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec), prec, rec)

def _selected_sets(model, p, z_idx, thr=1e-8):
    z = np.asarray(z_idx, int)
    beta = getattr(model, "beta_", None)
    if beta is None: beta = getattr(model, "beta", None)
    if beta is None: beta = getattr(model, "coef_", None)
    theta = getattr(model, "theta_", None)
    if theta is None: theta = getattr(model, "theta", None)
    mains = set()
    if beta is not None:
        b = np.asarray(beta, float).ravel()
        for j in range(min(p, b.size)):
            if abs(b[j]) > thr:
                mains.add(j)
    inter = set()
    if theta is not None:
        T = np.asarray(theta, float)
        for i in range(p):
            for jj in range(T.shape[1]):
                if abs(T[i, jj]) > thr:
                    inter.add(frozenset((i, int(z[jj]))))
    return mains, inter

def main(filename):
    # --- Extract parameters from filename ---
    stem = Path(filename).stem
    parts = stem.split("_")
    n_s, prev_s, p_s, rho_s = parts[-4:]
    n, prev, p, rho = int(n_s), float(prev_s), int(p_s), float(rho_s)

    # --- Read dataset ---
    df = pd.read_csv(filename)
    X = df[[f"X{i}" for i in range(1, p+1)]]
    y = df["y"]

    # --- Read truth file ---
    truth_file = str(Path(filename).with_name(
        Path(filename).name.replace("simulated_dataset_", "truth_simulated_dataset_").replace(".csv", ".json")
    ))
    # --- For full scale files, use this block ---
    # if not os.path.exists(truth_file):
    #     truth_file_alt = truth_file.replace("_full_scale.json", "_full_scale_full_scale.json")
    #     if os.path.exists(truth_file_alt):
    #         truth_file = truth_file_alt
    #     else:
    #         raise FileNotFoundError(f"Truth file not found for {filename}")
    if not os.path.exists(truth_file):
        raise FileNotFoundError(f"Truth file not found for {filename}")
    with open(truth_file) as f: t = json.load(f)
    Z_IDX  = t["interaction_core_idx"]
    M_TRUE = set(map(int, t["main_idx"]))
    I_TRUE = set(map(frozenset, t["interaction_pairs"]))

    # --- Nested CV for hyperparameter tuning (using PRIMARY_METRIC) ---
    y_bin = (y.values.ravel() > 0.5).astype(int)
    outer_cv = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=RANDOM_STATE)
    OUTER_ROWS, OUTER_PARAMS = [], []

    for k, (tr, te) in enumerate(outer_cv.split(X, y_bin), 1):
        Xtr, ytr = X.iloc[tr], y.iloc[tr]
        Xte, yte = X.iloc[te], y.iloc[te]
        inner_cv = StratifiedKFold(n_splits=N_INNER, shuffle=True, random_state=RANDOM_STATE + k)
        inner_scores = []
        for a in ALPHAS:
            for cvp in CVS:
                metric_scores = []
                for itr, ival in inner_cv.split(Xtr, (ytr.values.ravel() > 0.5).astype(int)):
                    Xi, yi = Xtr.iloc[itr], ytr.iloc[itr]
                    Xi_val, yi_val = Xtr.iloc[ival], ytr.iloc[ival]
                    y_train_bin = (yi.values.ravel() > 0.5).astype(int)
                    if len(np.unique(y_train_bin)) < 2:
                        continue
                    X_np = Xi.to_numpy(); Z = X_np[:, np.asarray(Z_IDX, int)]
                    m = PliableLasso(alpha=a,max_interaction_terms=1000, cv=cvp, eps=1e-4, normalize=True, verbose=False)
                    try:
                        m.fit(X_np, Z, yi.to_numpy().ravel())
                    except ValueError:
                        continue
                    X_val_np = Xi_val.to_numpy()
                    Z_val = X_val_np[:, np.asarray(Z_IDX, int)]
                    y_pred_val = m.predict(X_val_np, Z_val).ravel()
                    score = _score_metric(yi_val.values.ravel(), y_pred_val, PRIMARY_METRIC)
                    metric_scores.append(score)
                if not metric_scores:
                    continue
                inner_scores.append({
                    "alpha": a, "cv": cvp,
                    f"mean_{PRIMARY_METRIC}": float(np.mean(metric_scores)),
                    f"std_{PRIMARY_METRIC}": float(np.std(metric_scores)),
                })
        if not inner_scores:
            continue
        best = max(inner_scores, key=lambda r: r[f"mean_{PRIMARY_METRIC}"])
        X_np = Xtr.to_numpy(); Z = X_np[:, np.asarray(Z_IDX, int)]
        m = PliableLasso(alpha=best["alpha"], cv=best["cv"], eps=1e-4, normalize=True, verbose=False)
        try:
            m.fit(X_np, Z, ytr.to_numpy().ravel())
        except ValueError:
            continue
        OUTER_PARAMS.append((best["alpha"], best["cv"]))
        Xte_np = Xte.to_numpy()
        Zte = Xte_np[:, np.asarray(Z_IDX, int)]
        y_pred_te = m.predict(Xte_np, Zte).ravel()
        outer_metric = _score_metric(yte.values.ravel(), y_pred_te, PRIMARY_METRIC)
        OUTER_ROWS.append({
            "fold": k, "alpha": best["alpha"], "cv": best["cv"],
            f"outer_{PRIMARY_METRIC}": outer_metric
        })

    if not OUTER_ROWS:
        raise ValueError("No outer test results gathered.")

    NESTED_OUTER_RESULTS = pd.DataFrame(OUTER_ROWS).sort_values("fold").reset_index(drop=True)
    cnt = Counter(OUTER_PARAMS); top = max(cnt.values())
    cands = [pm for pm, c in cnt.items() if c == top]
    if len(cands) > 1:
        means = {
            pm: NESTED_OUTER_RESULTS.loc[
                    (NESTED_OUTER_RESULTS["alpha"] == pm[0]) &
                    (NESTED_OUTER_RESULTS["cv"] == pm[1]),
                    f"outer_{PRIMARY_METRIC}"
                ].mean()
            for pm in cands
        }
        chosen = max(means.items(), key=lambda kv: kv[1])[0]
    else:
        chosen = cands[0]
    NESTED_SUGGESTED_PARAMS = {"alpha": chosen[0], "cv": chosen[1]}

    # --- Refit on ALL data with the chosen params ---
    X_np = X.to_numpy()
    Z_all = X_np[:, np.asarray(Z_IDX, int)]
    best_model = PliableLasso(
        alpha=NESTED_SUGGESTED_PARAMS["alpha"],
        cv=NESTED_SUGGESTED_PARAMS["cv"],
        eps=1e-4,
        normalize=True,
        verbose=False
    )
    t0 = time()
    best_model.fit(X_np, Z_all, y.to_numpy().ravel())
    fit_time = time() - t0

    # --- Get predictions and metrics ---
    y_true = y.to_numpy().ravel()
    y_pred = best_model.predict(X_np, Z_all).ravel()
    y_pred_label = (y_pred > 0.5).astype(int)
    mse = mean_squared_error(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred_label)

    # --- Extract coefficients and selected sets ---
    beta = getattr(best_model, "beta_", None)
    if beta is None: beta = getattr(best_model, "beta", None)
    if beta is None: beta = getattr(best_model, "coef_", None)
    theta = getattr(best_model, "theta_", None)
    if theta is None: theta = getattr(best_model, "theta", None)
    intercept = getattr(best_model, "intercept_", None)
    b = np.asarray(beta, float).ravel() if beta is not None else np.zeros(p)
    T = np.asarray(theta, float) if theta is not None else np.zeros((p, len(Z_IDX)))
    mains = {j for j in range(min(p, b.size)) if abs(b[j]) > THR}
    z = np.asarray(Z_IDX, int)
    inter = set()
    gamma = {}
    for i in range(p):
        for jj in range(T.shape[1]):
            coef = T[i, jj]
            if abs(coef) > THR:
                inter.add(frozenset((i, int(z[jj]))))
                gamma[(i, int(z[jj]))] = float(coef)

    F1m, Pm, Rm = _f1(M_TRUE, mains)
    F1i, Pi, Ri = _f1(I_TRUE, inter)

    # --- Save results ---
    out_dir = os.path.join("results", "fit")
    # out_dir = os.path.join("results", "fit_predcv_full_scale")  # <-- for full scale
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir,
        f"best_model_simulated_dataset_{n}_{prev}_{p}_{rho}.json"
        # f"best_model_simulated_dataset_{n}_{prev}_{p}_{rho}_full_scale.json"  # <-- for full scale
    )
    save_dict = {
        "params": NESTED_SUGGESTED_PARAMS,
        "selected_mains": sorted(mains),
        "selected_interactions": [list(t) for t in gamma.keys()],
        "main_coefficients": {int(j): float(b[j]) for j in mains},
        "interaction_coefficients": {f"{i}_{j}": float(coef) for (i, j), coef in gamma.items()},
        "F1_main": F1m,
        "precision_main": Pm,
        "recall_main": Rm,
        "F1_inter": F1i,
        "precision_inter": Pi,
        "recall_inter": Ri,
        "fit_time_seconds": fit_time,
        "mse": mse,
        "accuracy": accuracy,
    }
    with open(out_path, "w") as f:
        json.dump(save_dict, f, indent=2)
    print(f"Saved best model summary to {out_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        data_dir = "C:/Users/enthe/Desktop/Thesis/data/simulated_data"
        for fname in os.listdir(data_dir):
            if (
                fname.startswith("simulated_dataset_")
                and fname.endswith(".csv")
                and not fname.endswith("_full_scale.csv")
                and "manifest" not in fname
            ):
                print(f"\n=== Running for {fname} ===")
                try:
                    main(os.path.join(data_dir, fname))
                except Exception as e:
                    print(f"Failed on {fname}: {e}")
        # --- For full scale files, use this block instead ---
        # for fname in os.listdir(data_dir):
        #     if (
        #         fname.startswith("simulated_dataset_")
        #         and fname.endswith("_full_scale.csv")
        #         and "manifest" not in fname
        #     ):
        #         print(f"\n=== Running for {fname} ===")
        #         try:
        #             main(os.path.join(data_dir, fname))
        #         except Exception as e:
        #             print(f"Failed on {fname}: {e}")