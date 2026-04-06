#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
— CPSS + noise-pmf over selection *frequency* (mains & interactions)
   + q = E[K] and conservative E[V] in summary.

MODIFIED:
  • Run CPSS ONCE to compute frequencies & diagnostics.
  • For each tau in [0.5, 0.6, 0.7], save all tau-dependent outputs
    (stable sets, metrics, bounds, overlap plots, summaries, etc.)
    into:
      C:\Users\enthe\Desktop\Thesis\results\stability_predcv_{tau}thresh
  • The frequency CSVs and tau-invariant diagnostic plots are also saved
    per-threshold folder (duplicated for convenience).
"""

import json
import os
import re
import argparse
from typing import Dict, Optional, Sequence, Set, Tuple, List
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plasso import PliableLasso
from sklearn.metrics import precision_recall_fscore_support

sns.set_theme(style="whitegrid", context="talk")

# ==========================
# Configuration (edit paths)
# ==========================

CONFIG = {
    "DATA_DIR": r"C:/Users/enthe/Desktop/Thesis/data/simulated_data",
    "RESULTS_BASE": r"C:/Users/enthe/Desktop/Thesis/results",
    "base_n": 500,
    "base_p": 20,
    "n_multipliers": (0.5, 1.0, 2.0),
    "prevalences": (0.05, 0.20, 0.50),
    "p_multipliers": (1, 25),
    "rhos": (0.0, 0.5, 0.8),
    "SKIP_MISSING": True,
    "B": 50,
    "cv": 0.1,
    "seed": 123,
    "mod_k": 10,
    "pair_aggregator": "max",
    "FIT_RESULTS_DIR": r"C:\Users\enthe\Desktop\Thesis\results\fit_predcv",
    "CHOSEN_JSON": None,
}

THRESH = 1e-5  # Only count as chosen/stable if abs(coef) >= THRESH

def _fmt_f(x: float) -> str:
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

def compute_metrics(stable, true, is_interaction=False):
    if is_interaction:
        stable = set(tuple(sorted(x)) for x in stable)
        true = set(tuple(sorted(x)) for x in true)
        all_items = stable | true
        y_true = [1 if item in true else 0 for item in all_items]
        y_pred = [1 if item in stable else 0 for item in all_items]
    else:
        all_items = stable | true
        y_true = [1 if item in true else 0 for item in all_items]
        y_pred = [1 if item in stable else 0 for item in all_items]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    TP = sum((p == 1 and t == 1) for p, t in zip(y_pred, y_true))
    FP = sum((p == 1 and t == 0) for p, t in zip(y_pred, y_true))
    FN = sum((p == 0 and t == 1) for p, t in zip(y_pred, y_true))
    n_stable = sum(y_pred)
    return dict(F1=f1, precision=precision, recall=recall, TP=TP, FP=FP, FN=FN, n_stable=n_stable)

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

def read_chosen_json(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)

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

def truth_sets(truth: Dict) -> Tuple[Set[int], Set[frozenset]]:
    mains = set(truth.get("main_idx", []))
    pairs = {frozenset(pair) for pair in truth.get("interaction_pairs", [])}
    return mains, pairs

def _stratified_complementary_halves(y: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y).reshape(-1)
    classes = np.unique(y)
    idx_by_class = {c: np.where(y == c)[0] for c in classes}
    half_A, half_B = [], []
    for _, idx_c in idx_by_class.items():
        perm = rng.permutation(idx_c)
        n_c = len(perm)
        kA = int(np.ceil(n_c / 2.0))
        A_c = perm[:kA]; B_c = perm[kA:]
        half_A.append(A_c); half_B.append(B_c)
    A = np.concatenate(half_A) if half_A else np.array([], dtype=int)
    B = np.concatenate(half_B) if half_B else np.array([], dtype=int)
    rng.shuffle(A); rng.shuffle(B)
    return A, B

def cpss_stability_selection(
    X: np.ndarray,
    y: np.ndarray,
    B_pairs: int = 50,
    cv: float = 0.1,
    random_state: int = 123,
    z_idx: Optional[Sequence[int]] = None,
    pair_aggregator: str = "max",
) -> Tuple[np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(random_state)
    n, p = X.shape
    z_idx = np.arange(p, dtype=int) if z_idx is None else np.asarray(z_idx, dtype=int)
    Z_full = X[:, z_idx]
    p_mod = Z_full.shape[1]

    mains_pair_count = np.zeros(p, dtype=float)
    inter_pair_count = np.zeros((p, p), dtype=float)
    K_eff = 0
    mains_half_sel_count = np.zeros(p, dtype=int)
    inter_half_sel_count = np.zeros((p, p), dtype=int)
    K_mains_halves: List[int] = []
    K_inter_halves: List[int] = []

    y = np.asarray(y).reshape(-1)
    if np.unique(y).size < 2:
        raise ValueError("y has a single class; CPSS needs both classes.")

    def fit_and_select(sub_idx: np.ndarray):
        Xb, yb = X[sub_idx], y[sub_idx]
        Zb = Z_full[sub_idx]
        mdl = PliableLasso(cv=cv,max_interaction_terms=1000, verbose=False, eps=1e-4, normalize=True)
        mdl.fit(Xb, Zb, yb)
        beta = getattr(mdl, "beta_", getattr(mdl, "beta", None))
        theta = getattr(mdl, "theta_", getattr(mdl, "theta", None))
        mains_sel = (np.abs(beta) >= THRESH) if beta is not None else np.zeros(p, dtype=bool)
        inter_sel = np.zeros((p, p), dtype=bool)
        if theta is not None:
            sel = (np.abs(theta) >= THRESH)
            for i in range(p):
                for jj in range(p_mod):
                    if sel[i, jj]:
                        j = int(z_idx[jj])
                        inter_sel[i, j] = True
                        inter_sel[j, i] = True
        return mains_sel, inter_sel

    for _ in range(B_pairs):
        A_idx, B_idx = _stratified_complementary_halves(y, rng)
        if np.unique(y[A_idx]).size < 2 or np.unique(y[B_idx]).size < 2:
            continue
        try:
            mains_A, inter_A = fit_and_select(A_idx)
            mains_B, inter_B = fit_and_select(B_idx)
        except Exception:
            continue

        K_mains_halves.extend([int(mains_A.sum()), int(mains_B.sum())])

        def count_pairs(M):
            ct = 0
            for i in range(p):
                for j in range(i, p):
                    if M[i, j]:
                        ct += 1
            return ct
        K_inter_halves.extend([count_pairs(inter_A), count_pairs(inter_B)])

        mains_half_sel_count += mains_A.astype(int)
        mains_half_sel_count += mains_B.astype(int)
        for i in range(p):
            for j in range(i, p):
                if inter_A[i, j]:
                    inter_half_sel_count[i, j] += 1
                if inter_B[i, j]:
                    inter_half_sel_count[i, j] += 1

        if pair_aggregator == "max":
            mains_pair = np.logical_or(mains_A, mains_B)
            inter_pair = np.logical_or(inter_A, inter_B)
        elif pair_aggregator == "mean":
            mains_pair = ((mains_A.astype(int) + mains_B.astype(int)) / 2.0) >= 0.5
            inter_pair = ((inter_A.astype(int) + inter_B.astype(int)) / 2.0) >= 0.5
        elif pair_aggregator == "min":
            mains_pair = np.logical_and(mains_A, mains_B)
            inter_pair = np.logical_and(inter_A, inter_B)
        else:
            raise ValueError("pair_aggregator must be one of {'max','mean','min'}")

        mains_pair_count += mains_pair.astype(float)
        inter_pair_count += inter_pair.astype(float)
        K_eff += 1

    if K_eff == 0:
        raise RuntimeError("All CPSS pairs failed; try adjusting parameters or random seed.")

    mains_freq = mains_pair_count / K_eff
    inter_freq = inter_pair_count / K_eff

    extras = {
        "K_mains_halves": K_mains_halves,
        "K_inter_halves": K_inter_halves,
        "K_eff": K_eff,
        "pair_aggregator": pair_aggregator,
        "mains_counts_per_var": mains_pair_count.astype(int).tolist(),
        "mains_half_sel_count": mains_half_sel_count.astype(int).tolist(),
        "inter_counts_upper": [
            {"i": i, "j": j, "count": int(inter_pair_count[i, j])}
            for i in range(p) for j in range(i, p)
        ],
        "inter_half_counts_upper": [
            {"i": i, "j": j, "half_count": int(inter_half_sel_count[i, j])}
            for i in range(p) for j in range(i, p)
        ],
    }
    return mains_freq, inter_freq, extras

# ...rest of the plotting and utility functions unchanged...

def save_all_for_tau(
    tau: float,
    results_dir: str,
    stem: str,
    p: int,
    mains_true: Set[int],
    inter_true: Set[frozenset],
    mains_freq: np.ndarray,
    inter_freq: np.ndarray,
    extras: dict,
    chosen_json_path: Optional[str]
):
    ensure_dir(results_dir)

    # Stable (freq >= tau)
    stable_mains = set(int(i) for i in np.where(mains_freq >= tau)[0])
    stable_interactions = {
        frozenset((int(i), int(j)))
        for i in range(p) for j in range(i, p)
        if inter_freq[i, j] >= tau
    }

    # Save per-feature / per-pair frequencies (duplicated per tau folder)
    mains_freq_path = os.path.join(results_dir, f"stability_mains_{stem}.csv")
    inter_freq_path = os.path.join(results_dir, f"stability_interactions_{stem}.csv")
    pd.DataFrame({"freq": mains_freq}).to_csv(mains_freq_path, index=False)
    rows = []
    for i in range(p):
        for j in range(i, p):
            rows.append({"i": i, "j": j, "freq": float(inter_freq[i, j])})
    pd.DataFrame(rows).to_csv(inter_freq_path, index=False)
    print(f"[{stem} | τ={tau}] Saved CPSS frequencies → mains:{mains_freq_path} | inter:{inter_freq_path}")

    # ------ Noise PMFs (selection frequency)
    all_idx = set(range(p))
    noise_mains_idx = sorted(all_idx - mains_true)
    noise_inter_pairs = [(i, j) for i in range(p) for j in range(i, p)
                         if frozenset((i, j)) not in inter_true]

    m_counts_all = np.array(extras["mains_counts_per_var"], dtype=int)
    noise_m_counts = m_counts_all[noise_mains_idx] if len(noise_mains_idx) else np.array([], dtype=int)
    inter_counts_map = {(d["i"], d["j"]): d["count"] for d in extras["inter_counts_upper"]}
    noise_i_counts = np.array([inter_counts_map[(i, j)] for (i, j) in noise_inter_pairs], dtype=int)

    mains_metrics = compute_metrics(stable_mains, mains_true, is_interaction=False)
    interactions_metrics = compute_metrics(stable_interactions, inter_true, is_interaction=True)

    def pmf_over_counts(counts, B_pairs):
        support_counts = np.arange(0, B_pairs + 1, dtype=int)
        hist = np.bincount(counts, minlength=B_pairs + 1).astype(float) if counts.size else np.zeros(B_pairs + 1)
        pmf = hist / hist.sum() if hist.sum() > 0 else hist
        return support_counts.tolist(), pmf.tolist()

    B_pairs = int(extras["K_eff"])
    sup_m_counts, pmf_m_noise = pmf_over_counts(noise_m_counts, B_pairs)
    sup_i_counts, pmf_i_noise = pmf_over_counts(noise_i_counts, B_pairs)

    sup_m_freq = [c / B_pairs for c in sup_m_counts]
    sup_i_freq = [c / B_pairs for c in sup_i_counts]

    r_val = -0.5
    uni_m_noise = is_unimodal(pmf_m_noise); r_m_noise = r_concave_ok(pmf_m_noise, r=r_val)
    uni_i_noise = is_unimodal(pmf_i_noise); r_i_noise = r_concave_ok(pmf_i_noise, r=r_val)

    true_main_freqs = sorted([float(mains_freq[i]) for i in sorted(mains_true)]) if mains_true else []
    true_inter_freqs = []
    if inter_true:
        for pair in sorted(inter_true):
            i, j = sorted(tuple(pair))
            true_inter_freqs.append(float(inter_freq[i, j]))

    pmf_noise_mains_png = os.path.join(results_dir, f"pmf_noise_freq_mains_{stem}.png")
    pmf_noise_inter_png = os.path.join(results_dir, f"pmf_noise_freq_interactions_{stem}.png")
    plot_selection_pmf_freq_with_lines(
        sup_m_freq, pmf_m_noise,
        f"{stem} — Noise mains selection-frequency PMF",
        pmf_noise_mains_png, uni_m_noise, r_m_noise, r_val,
        line_freqs=true_main_freqs, line_label="True main frequency"
    )
    plot_selection_pmf_freq_with_lines(
        sup_i_freq, pmf_i_noise,
        f"{stem} — Noise interactions selection-frequency PMF",
        pmf_noise_inter_png, uni_i_noise, r_i_noise, r_val,
        line_freqs=true_inter_freqs, line_label="True interaction frequency"
    )

    mains_pmf_info = empirical_pmf(extras["K_mains_halves"])
    inter_pmf_info = empirical_pmf(extras["K_inter_halves"])
    q_main = mains_pmf_info["q"]; q_inter = inter_pmf_info["q"]

    uni_main = is_unimodal(mains_pmf_info["pmf"]); r_main = r_concave_ok(mains_pmf_info["pmf"], r=r_val)
    uni_inter = is_unimodal(inter_pmf_info["pmf"]); r_inter = r_concave_ok(inter_pmf_info["pmf"], r=r_val)

    pmf_q_main_png = os.path.join(results_dir, f"pmf_q_mains_{stem}.png")
    pmf_q_inter_png = os.path.join(results_dir, f"pmf_q_interactions_{stem}.png")
    plot_pmf_q(mains_pmf_info["support"], mains_pmf_info["pmf"],
               f"{stem} — PMF of K per half (MAINS) for q estimation",
               pmf_q_main_png, uni_main, r_main, r_val)
    plot_pmf_q(inter_pmf_info["support"], inter_pmf_info["pmf"],
               f"{stem} — PMF of K per half (INTERACTIONS) for q estimation",
               pmf_q_inter_png, uni_inter, r_inter, r_val)

    denom_halves = 2 * int(extras["K_eff"])
    mains_half = np.asarray(extras["mains_half_sel_count"], dtype=int)
    mains_half_highlight_png = os.path.join(results_dir, f"mains_perhalf_freq_highlight_{stem}.png")
    plot_mains_perhalf_freq_highlight(
        mains_half, mains_true, denom_halves, mains_half_highlight_png, stem
    )

    p_inter_universe = (p * (p + 1)) // 2
    EV_main = conservative_false_selection_bound(q_main, tau, p)
    EV_inter = conservative_false_selection_bound(q_inter, tau, p_inter_universe)

    # --- Load chosen results (if available), filter by abs(coef) >= THRESH
    chosen_mains = set()
    chosen_interactions = set()
    if chosen_json_path and os.path.exists(chosen_json_path):
        with open(chosen_json_path, "r") as f:
            chosen_data = json.load(f)
            # Filter chosen mains by abs(coef) >= THRESH
            for idx, coef in chosen_data.get("main_coefficients", {}).items():
                if abs(coef) >= THRESH:
                    chosen_mains.add(int(idx))
            # Filter chosen interactions by abs(coef) >= THRESH
            for key, coef in chosen_data.get("interaction_coefficients", {}).items():
                i, j = map(int, key.split("_"))
                if abs(coef) >= THRESH:
                    chosen_interactions.add(frozenset((i, j)))

    # --- Overlap counts (mains)
    chosen_and_stable = chosen_mains & stable_mains
    chosen_and_true = chosen_mains & mains_true
    stable_and_true = stable_mains & mains_true
    all_three = chosen_mains & stable_mains & mains_true

    mains_counts = {
        "chosen_total": len(chosen_mains),
        "stable_total": len(stable_mains),
        "true_total": len(mains_true),
        "chosen_and_stable": len(chosen_and_stable),
        "chosen_and_true": len(chosen_and_true),
        "stable_and_true": len(stable_and_true),
        "all_three": len(all_three),
    }

    # --- Overlap counts (interactions)
    chosen_and_stable_inter = chosen_interactions & stable_interactions
    chosen_and_true_inter = chosen_interactions & inter_true
    stable_and_true_inter = stable_interactions & inter_true
    all_three_inter = chosen_interactions & stable_interactions & inter_true

    inter_counts = {
        "chosen_total": len(chosen_interactions),
        "stable_total": len(stable_interactions),
        "true_total": len(inter_true),
        "chosen_and_stable": len(chosen_and_stable_inter),
        "chosen_and_true": len(chosen_and_true_inter),
        "stable_and_true": len(stable_and_true_inter),
        "all_three": len(all_three_inter),
    }

    labels = ["Chosen", "Stable", "True", "Chosen & Stable", "Chosen ∩ True", "Stable ∩ True", "All Three"]
    counts_m = [
        mains_counts["chosen_total"], mains_counts["stable_total"], mains_counts["true_total"],
        mains_counts["chosen_and_stable"], mains_counts["chosen_and_true"],
        mains_counts["stable_and_true"], mains_counts["all_three"]
    ]
    counts_i = [
        inter_counts["chosen_total"], inter_counts["stable_total"], inter_counts["true_total"],
        inter_counts["chosen_and_stable"], inter_counts["chosen_and_true"],
        inter_counts["stable_and_true"], inter_counts["all_three"]
    ]

    barplot_mains_png = os.path.join(results_dir, f"barplot_mains_{stem}.png")
    barplot_inters_png = os.path.join(results_dir, f"barplot_interactions_{stem}.png")
    plot_overlap_bars(labels, counts_m, f"{stem} — Main Effects: Chosen, Stable, True, Overlaps", barplot_mains_png)
    plot_overlap_bars(labels, counts_i, f"{stem} — Interactions: Chosen, Stable, True, Overlaps", barplot_inters_png)

    summary = {
        "dataset_stem": stem,
        "tau": tau,
        "q_estimates": {"mains": q_main, "interactions": q_inter},
        "E[V]_bounds_conservative": {
            "mains": EV_main,
            "interactions": EV_inter,
            "note": "Classic CPSS-style conservative bound using q=E[K] and 1/(2*tau-1)."
        },
        "noise_pmf_frequency": {
            "B_pairs": B_pairs,
            "mains": {"freq_grid": sup_m_freq, "pmf": pmf_m_noise,
                      "unimodal": uni_m_noise, "r": r_val, "r_concave_ok": r_m_noise,
                      "true_main_freqs": true_main_freqs},
            "interactions": {"freq_grid": sup_i_freq, "pmf": pmf_i_noise,
                             "unimodal": uni_i_noise, "r": r_val, "r_concave_ok": r_i_noise,
                             "true_inter_freqs": true_inter_freqs}
        },
        "mains_counts": mains_counts,
        "interactions_counts": inter_counts,
        "mains_metrics": mains_metrics,
        "interactions_metrics": interactions_metrics,
        "files": {
            "mains_freq_csv": mains_freq_path,
            "inter_freq_csv": inter_freq_path,
            "pmf_noise_freq_mains_png": pmf_noise_mains_png,
            "pmf_noise_freq_interactions_png": pmf_noise_inter_png,
            "pmf_q_mains_png": pmf_q_main_png,
            "pmf_q_interactions_png": pmf_q_inter_png,
            "mains_perhalf_freq_highlight_png": mains_half_highlight_png,
            "barplot_mains_png": barplot_mains_png,
            "barplot_interactions_png": barplot_inters_png,
        },
        "stable_indices": {
            "mains": [int(x) for x in sorted(stable_mains)],
            "interactions": [[int(i) for i in sorted(list(pair))] for pair in stable_interactions]
        }
    }
    summary_path = os.path.join(results_dir, f"stability_summary_{stem}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{stem} | τ={tau}] Saved summary → {summary_path}")

    extended = {
        "dataset_stem": stem,
        "final_tau": tau,
        "pair_aggregator": extras.get("pair_aggregator"),
        "K_eff_pairs": int(extras["K_eff"]),
        "pmf_K_per_half": {
            "mains": mains_pmf_info,
            "interactions": inter_pmf_info,
            "shape_checks": {
                "mains": {"unimodal": uni_main, "r": r_val, "r_concave_ok": r_main},
                "interactions": {"unimodal": uni_inter, "r": r_val, "r_concave_ok": r_inter},
            }
        },
        "noise_selection_pmf_frequency": summary["noise_pmf_frequency"],
        "bounds": {
            "conservative": {
                "tau": tau, "q_mains": q_main, "q_interactions": q_inter,
                "E[V]_mains": EV_main, "E[V]_interactions": EV_inter,
                "p_mains": p, "p_interactions_universe": (p * (p + 1)) // 2
            }
        },
        "files": summary["files"] | {"summary_json": summary_path},
    }
    ext_path = os.path.join(results_dir, f"stability_extended_{stem}.json")
    with open(ext_path, "w") as f:
        json.dump(extended, f, indent=2)
    print(f"[{stem} | τ={tau}] Saved extended diagnostics → {ext_path}")

    print(
        f"[{stem}] τ={tau:.3f} | q_mains≈{q_main:.2f}, q_inter≈{q_inter:.2f} | "
        f"E[V]_mains≤{EV_main:.2f}, E[V]_inter≤{EV_inter:.2f} (conservative)"
    )
# -------------------------
# PMF, q, checks
# -------------------------

def empirical_pmf(counts: List[int]) -> Dict[str, object]:
    if len(counts) == 0:
        return {"support": [], "pmf": [], "q": 0.0}
    counts = np.asarray(counts, dtype=int)
    support = np.arange(0, counts.max() + 1, dtype=int)
    hist = np.bincount(counts, minlength=support.size).astype(float)
    pmf = hist / hist.sum()
    q = float(np.dot(support, pmf))
    return {"support": support.tolist(), "pmf": pmf.tolist(), "q": q}

def is_unimodal(pmf: List[float], tol: float = 1e-12) -> bool:
    p = np.asarray(pmf, dtype=float)
    if p.sum() <= 0: return False
    idx_mode = int(np.argmax(p))
    left = np.all(np.diff(p[:idx_mode+1]) >= -tol)
    right = np.all(np.diff(p[idx_mode:]) <= tol)
    return bool(left and right)

def r_concave_ok(pmf: List[float], r: float = -0.5, tol: float = 1e-10) -> bool:
    p = np.asarray(pmf, dtype=float)
    mask = p > 0
    if mask.sum() < 3:
        return True
    x = p[mask] ** r
    second = x[:-2] - 2 * x[1:-1] + x[2:]
    return bool(np.all(second <= tol))

# -------------------------
# Bounds (conservative)
# -------------------------

def conservative_false_selection_bound(q: float, tau: float, p: int) -> float:
    if tau <= 0.5:
        return float("inf")
    return q**2 / (p * max(1e-12, (2.0 * tau - 1.0)))

# -------------------------
# Plots
# -------------------------

def plot_pmf_q(support: List[int], pmf: List[float], title: str, out_png: str,
               uni_ok: bool, r_ok: bool, r_val: float):
    plt.figure(figsize=(8,5))
    ax = sns.barplot(x=[str(k) for k in support], y=pmf)
    ax.set_title(f"{title}\n(Unimodal={uni_ok} | r-concave r={r_val}: {r_ok})")
    ax.set_xlabel("K selected per half"); ax.set_ylabel("Empirical pmf")
    plt.tight_layout(); plt.savefig(out_png, dpi=220); plt.close()

def _sparse_tick_labels(ax, freq_grid: List[float], max_labels: int = 10):
    labels = [f"{x:.2f}".rstrip('0').rstrip('.') for x in freq_grid]
    ax.set_xticklabels(labels, rotation=0)
    n = len(labels)
    if n <= max_labels:
        return
    step = int(np.ceil(n / max_labels))
    for i, tick in enumerate(ax.get_xticklabels()):
        tick.set_visible(i % step == 0)

def plot_selection_pmf_freq_with_lines(
    freq_grid: List[float],
    pmf: List[float],
    title: str,
    out_png: str,
    uni_ok: bool,
    r_ok: bool,
    r_val: float,
    line_freqs: Optional[List[float]] = None,
    line_label: Optional[str] = None,
):
    plt.figure(figsize=(10,5))
    labels = [f"{x:.2f}".rstrip('0').rstrip('.') for x in freq_grid]
    ax = sns.barplot(x=labels, y=pmf)

    if line_freqs:
        from matplotlib.lines import Line2D
        added = False
        for f in line_freqs:
            xlab = f"{f:.2f}".rstrip('0').rstrip('.')
            ax.axvline(x=xlab, linestyle="--", linewidth=2, color="black", alpha=0.9)
            added = True
        if added and line_label:
            proxy = Line2D([0], [0], linestyle="--", color="black", lw=2, label=line_label)
            ax.legend(handles=[proxy], loc="upper right", frameon=True)

    _sparse_tick_labels(ax, freq_grid, max_labels=10)
    ax.set_title(f"{title}\n(Unimodal={uni_ok} | r-concave r={r_val}: {r_ok})")
    ax.set_xlabel("Selection frequency (across pairs)"); ax.set_ylabel("Empirical pmf")
    plt.tight_layout(); plt.savefig(out_png, dpi=220); plt.close()

def plot_mains_perhalf_freq_highlight(
    mains_half_sel_count: np.ndarray,
    true_mains: Set[int],
    denom_halves: int,
    out_png: str,
    stem: str
):
    """
    One bar per MAIN feature: (times selected across halves) / (2*K_eff).
    True mains colored.
    """
    p = len(mains_half_sel_count)
    freq = np.array(mains_half_sel_count, dtype=float) / max(1, denom_halves)
    x = list(range(p))
    palette = ["#ff7f0e" if i in true_mains else "#1f77b4" for i in x]
    plt.figure(figsize=(12,5))
    ax = sns.barplot(x=x, y=freq, palette=palette)
    ax.set_title(f"{stem} — Mains selection frequency per half (per feature)")
    ax.set_xlabel("Feature index"); ax.set_ylabel("Frequency over halves")
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#ff7f0e", edgecolor="black", label="True main"),
        Patch(facecolor="#1f77b4", edgecolor="black", label="Other"),
    ], loc="upper right", frameon=True)
    plt.tight_layout(); plt.savefig(out_png, dpi=220); plt.close()

def plot_overlap_bars(labels: List[str], counts: List[int], title: str, out_png: str):
    plt.figure(figsize=(9,7))
    ax = sns.barplot(x=labels, y=counts, hue=labels, palette="muted", legend=False)
    ax.set_title(title)
    ax.set_ylabel("Count"); ax.set_xlabel("")
    ax.set_xticklabels(labels, rotation=20, ha="right")
    for i, v in enumerate(counts):
        ax.text(i, v + 0.2, str(v), ha="center", va="bottom", fontsize=12)
    plt.tight_layout(); plt.savefig(out_png, dpi=220); plt.close()

def plot_interaction_prob_hist(inter_freq: np.ndarray,
                               true_pairs: Set[frozenset],
                               out_png: str):
    p = inter_freq.shape[0]
    vals = []; marks = []
    for i in range(p):
        for j in range(i, p):
            v = inter_freq[i, j]
            vals.append(v)
            if frozenset((i, j)) in true_pairs:
                marks.append(v)
    vals = np.asarray(vals, dtype=float)
    plt.figure(figsize=(9,5))
    ax = plt.gca()
    ax.hist(vals, bins=20, alpha=0.85, edgecolor="black")
    for v in marks:
        ax.axvline(v, linestyle="--", linewidth=2)
    ax.set_title("Interaction selection probabilities (upper-tri incl. diag)")
    ax.set_xlabel("CPSS frequency"); ax.set_ylabel("Count")
    if marks:
        ax.text(0.02, 0.95, f"{len(marks)} true interactions marked",
                transform=ax.transAxes, va="top")
    plt.tight_layout(); plt.savefig(out_png, dpi=220); plt.close()
    
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def per_tau_results_dir(tau: float) -> str:
    # Windows-safe formatting: "0.5" -> "0.5", "0.60" -> "0.6"
    tau_str = str(float(tau))
    return os.path.join(CONFIG["RESULTS_BASE"], f"stability_predcv_{tau_str}thresh")

def run_single_dataset(csv_path: str, tau_list: Sequence[float]):
    try:
        csv_path, truth_path, out_stem_default = parse_sim_filename(csv_path)
        if not os.path.exists(csv_path): raise FileNotFoundError(csv_path)
        if not os.path.exists(truth_path): raise FileNotFoundError(truth_path)

        stem = os.path.basename(out_stem_default)
        fit_dir = CONFIG.get("FIT_RESULTS_DIR")
        chosen_json_path = CONFIG["CHOSEN_JSON"] or (os.path.join(fit_dir, f"best_model_{stem}.json") if fit_dir else None)

        X, y = read_sim_csv(csv_path)
        truth = read_truth_json(truth_path)
        mains_true, inter_true = truth_sets(truth)
        p = X.shape[1]
        z_idx = derive_z_indices(truth, X, k_if_no_truth=int(CONFIG["mod_k"]))

        mains_freq, inter_freq, extras = cpss_stability_selection(
            X, y,
            B_pairs=int(CONFIG["B"]),
            cv=float(CONFIG["cv"]),
            random_state=int(CONFIG["seed"]),
            z_idx=z_idx,
            pair_aggregator=str(CONFIG.get("pair_aggregator", "max")),
        )

        for tau in tau_list:
            results_dir = per_tau_results_dir(tau)
            save_all_for_tau(
                tau=float(tau),
                results_dir=results_dir,
                stem=stem,
                p=p,
                mains_true=mains_true,
                inter_true=inter_true,
                mains_freq=mains_freq,
                inter_freq=inter_freq,
                extras=extras,
                chosen_json_path=chosen_json_path
            )

        return True, stem, None
    except Exception as e:
        return False, os.path.basename(os.path.splitext(csv_path)[0]), str(e)

def main():
    parser = argparse.ArgumentParser(description="CPSS once + per-threshold saving of stability outputs")
    parser.add_argument("--csv", type=str, default=None,
                        help="Run a single dataset CSV (file-by-file). If omitted, run batch over CONFIG.DATA_DIR.")
    parser.add_argument("--taus", type=str, default="0.5,0.55,0.6,0.65,0.7",
                        help="Comma-separated list of tau thresholds (default: 0.5,0.55,0.6,0.65,0.7).")
    args = parser.parse_args()

    tau_list = [float(t.strip()) for t in args.taus.split(",") if t.strip()]

    if args.csv:
        print(f"[INFO] Single-dataset mode: {args.csv}")
        ok, s, err = run_single_dataset(args.csv, tau_list=tau_list)
        if not ok:
            print(f"[ERROR] {s}: {err}")
        return

    csv_paths = dataset_paths_from_config(CONFIG)
    print(f"[INFO] Planned datasets: {len(csv_paths)}")

    successes, failures = [], []
    for path in csv_paths:
        stem = os.path.basename(path)[:-4]
        truth_path = os.path.join(os.path.dirname(path), f"truth_{stem}.json")
        missing = []
        if not os.path.exists(path): missing.append("csv")
        if not os.path.exists(truth_path): missing.append("truth")
        if missing:
            msg = f"missing {'/'.join(missing)} for {stem}"
            if CONFIG["SKIP_MISSING"]:
                print(f"[SKIP] {msg}")
                failures.append((stem, msg))
                continue
            else:
                raise FileNotFoundError(msg)

        print(f"\n================= CPSS Stability (one run) → per-τ saves: {stem} =================")
        ok, s, err = run_single_dataset(path, tau_list=tau_list)
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