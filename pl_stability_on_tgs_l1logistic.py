import os 
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIG ---
CONFIG = {
    "DATA_PATH": "C:/Users/enthe/Desktop/Thesis/data/tgs_data/tgs_dataset_normalized.csv",
    "RESULTS_DIR": "C:/Users/enthe/Desktop/Thesis/results/tgs_results",
    "B": 50,
    "cv": 0.1,
    "seed": 123,
    "mod_k": 10,
    "logreg_cs": np.logspace(-4, 1, 20),
}

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def read_tgs_csv(path):
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    y = pd.to_numeric(df["MCI"], errors="coerce").to_numpy(dtype=float)
    x_cols = [c for c in df.columns if c != "MCI" and pd.api.types.is_numeric_dtype(df[c])]
    X = df[x_cols].apply(pd.to_numeric, errors="coerce")
    return X.to_numpy(dtype=float), y.astype(int), x_cols

def select_z_l1logistic(X, y, x_cols, max_k=10, Cs=None, seed=13):
    logreg = LogisticRegressionCV(
        Cs=Cs, cv=5, penalty="l1", solver="liblinear", random_state=seed, max_iter=1000
    )
    logreg.fit(X, y)
    coefs = np.abs(logreg.coef_).ravel()
    main_indices = np.where(coefs > 1e-8)[0]
    if len(main_indices) > max_k:
        sorted_idx = main_indices[np.argsort(-coefs[main_indices])]
        main_indices = sorted_idx[:max_k]
    return main_indices

def empirical_pmf(counts):
    """
    Empirical pmf of integer counts: K ∈ {0,1,...}.
    Returns support, pmf, and q = E[K].
    """
    if len(counts) == 0:
        return {"support": [], "pmf": [], "q": 0.0}
    counts = np.asarray(counts, dtype=int)
    support = np.arange(0, counts.max() + 1, dtype=int)
    hist = np.bincount(counts, minlength=support.size).astype(float)
    pmf = hist / hist.sum()
    q = float(np.dot(support, pmf))
    return {"support": support.tolist(), "pmf": pmf.tolist(), "q": q}

def is_unimodal(pmf, tol=1e-12):
    p = np.asarray(pmf, float)
    if p.sum() <= 0:
        return False
    idx_mode = int(np.argmax(p))
    left = np.all(np.diff(p[:idx_mode+1]) >= -tol)
    right = np.all(np.diff(p[idx_mode:]) <= tol)
    return bool(left and right)

def r_concave_ok(pmf, r=-0.5, tol=1e-10):
    """
    Check r-concavity: for r < 0, f is r-concave if f^r is convex.
    Discrete convexity: second differences >= 0.
    """
    p = np.asarray(pmf, float)
    mask = p > 0
    if mask.sum() < 3:
        return True
    x = p[mask] ** r
    second = x[:-2] - 2 * x[1:-1] + x[2:]
    return bool(np.all(second >= -tol))  # convex up to small numerical tolerance

def conservative_false_selection_bound(q, tau, p):
    """
    Worst-case CPSS bound (Theorem 1-style):
    E[V] <= q^2 / (p * (2*tau - 1)) for tau > 0.5.
    """
    if tau <= 0.5:
        return float("inf")
    return q**2 / (p * max(1e-12, (2.0 * tau - 1.0)))

def plot_pmf(support, pmf, title, out_png, uni_ok, r_ok, r_val):
    plt.figure(figsize=(8,5))
    ax = sns.barplot(x=[str(k) for k in support], y=pmf)
    ax.set_title(f"{title}\nUnimodal={uni_ok} | r-concave(r={r_val})={r_ok}")
    ax.set_xlabel("K (selected per half)")
    ax.set_ylabel("Empirical pmf")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

# --- NEW: build pmf from probabilities Π ∈ {0,1/B_eff,...,1} ---
def pmf_from_probs(probs, B_eff):
    """
    Map probabilities Π in {0,1/B_eff,...,1} to counts j ∈ {0,...,B_eff},
    then use empirical_pmf on those counts.
    """
    if B_eff <= 0 or len(probs) == 0:
        return np.array([]), np.array([])
    probs = np.asarray(probs, float)
    counts = np.rint(probs * B_eff).astype(int)
    counts[counts < 0] = 0
    counts[counts > B_eff] = B_eff
    res = empirical_pmf(counts)
    return np.array(res["support"], dtype=int), np.array(res["pmf"], dtype=float)

def cpss_stability_selection_tgs(X, y, B_pairs=50, Cs=None, seed=123):
    rng = np.random.default_rng(seed)
    n, p = X.shape

    # ---- Build all pairwise interactions for ALL p features ----
    inter_cols = []
    inter_pairs = []   # list of (i, j) with i < j
    for i in range(p):
        for j in range(i + 1, p):
            inter_pairs.append((i, j))
            inter_cols.append(X[:, i] * X[:, j])

    if inter_cols:
        Inter = np.column_stack(inter_cols)  # (n, n_inter)
    else:
        Inter = np.zeros((n, 0), dtype=float)
    n_inter = Inter.shape[1]

    # --- counts for CPSS frequencies (mains and interactions) ---
    mains_pair_or_count = np.zeros(p, dtype=float)   # CPSS freq: selected in at least one half
    mains_pair_and_count = np.zeros(p, dtype=float)  # NEW: selected in both halves (Π̃)
    mains_half_count = np.zeros(p, dtype=float)      # NEW: total #halves selecting each var (for Π̂)

    inter_pair_count = np.zeros((p, p), dtype=float)
    K_eff = 0
    K_mains_halves = []
    K_inter_halves = []

    def fit_and_select(sub_idx):
        Xb_main = X[sub_idx]                             # (n_sub, p)
        Xb_inter = Inter[sub_idx] if n_inter > 0 else np.empty((len(sub_idx), 0))
        Xb = np.hstack([Xb_main, Xb_inter])              # mains + interactions
        yb = y[sub_idx]

        logreg = LogisticRegressionCV(
            Cs=Cs,
            cv=5,
            penalty="l1",
            solver="liblinear",
            random_state=seed,
            max_iter=1000,
        )
        logreg.fit(Xb, yb)
        beta = logreg.coef_.ravel()

        # Split coefficients into mains and interactions
        beta_main = beta[:p]
        beta_inter = beta[p:]

        mains_sel = (np.abs(beta_main) > 1e-8)

        inter_sel = np.zeros((p, p), dtype=bool)
        k_inter_half = 0
        if n_inter > 0:
            inter_sel_flat = (np.abs(beta_inter) > 1e-8)  # length n_inter
            for k, (i, j) in enumerate(inter_pairs):
                if inter_sel_flat[k]:
                    inter_sel[i, j] = True
                    inter_sel[j, i] = True
                    k_inter_half += 1

        k_mains_half = int(mains_sel.sum())
        return mains_sel, inter_sel, k_mains_half, k_inter_half

    # ---- CPSS loop over complementary pairs ----
    for _ in range(B_pairs):
        # Stratified complementary halves
        classes = np.unique(y)
        idx_by_class = {c: np.where(y == c)[0] for c in classes}
        half_A, half_B = [], []
        for c, idx_c in idx_by_class.items():
            perm = rng.permutation(idx_c)
            n_c = len(perm)
            kA = int(np.ceil(n_c / 2.0))
            A_c = perm[:kA]
            B_c = perm[kA:]
            half_A.append(A_c)
            half_B.append(B_c)
        A = np.concatenate(half_A) if half_A else np.array([], dtype=int)
        B = np.concatenate(half_B) if half_B else np.array([], dtype=int)
        rng.shuffle(A)
        rng.shuffle(B)

        # Need both classes in each half
        if np.unique(y[A]).size < 2 or np.unique(y[B]).size < 2:
            continue

        try:
            mains_A, inter_A, k_main_A, k_inter_A = fit_and_select(A)
            mains_B, inter_B, k_main_B, k_inter_B = fit_and_select(B)
            K_mains_halves.extend([k_main_A, k_main_B])
            K_inter_halves.extend([k_inter_A, k_inter_B])
        except Exception as e:
            print(f"Fit failed in one CPSS half: {e}")
            continue

        # per-half selection counts for Π̂_B
        mains_half_count += mains_A.astype(float)
        mains_half_count += mains_B.astype(float)

        # union (CPSS frequency) and intersection (Π̃_B)
        mains_pair_or = np.logical_or(mains_A, mains_B)
        mains_pair_and = np.logical_and(mains_A, mains_B)

        mains_pair_or_count += mains_pair_or.astype(float)
        mains_pair_and_count += mains_pair_and.astype(float)

        inter_pair = np.logical_or(inter_A, inter_B)
        inter_pair_count += inter_pair.astype(float)

        K_eff += 1

    if K_eff == 0:
        raise RuntimeError("All CPSS pairs failed; try adjusting parameters or random seed.")

    # --- Frequencies used for CPSS (mains/interactions) ---
    mains_freq = mains_pair_or_count / K_eff    # freq of appearing in at least one half
    inter_freq = inter_pair_count / K_eff

    # --- q estimates (per half) ---
    K_mains_halves_arr = np.array(K_mains_halves, dtype=float)
    K_inter_halves_arr = np.array(K_inter_halves, dtype=float)
    if K_mains_halves_arr.size == 0:
        q_mains = 0.0
    else:
        q_mains = float(K_mains_halves_arr.mean())
    if K_inter_halves_arr.size == 0:
        q_inter = 0.0
    else:
        q_inter = float(K_inter_halves_arr.mean())
    if K_mains_halves_arr.size == 0:
        q_total = 0.0
    else:
        q_total = float((K_mains_halves_arr + K_inter_halves_arr).mean())

    # --- Π̂_B(k) and Π̃_B(k) (mains only) ---
    Pi_hat = mains_half_count / (2.0 * K_eff)       # proportion of halves selecting each main
    Pi_tilde = mains_pair_and_count / K_eff         # proportion of pairs selecting in both halves

    q_mains_sum = float(Pi_hat.sum())               # should ≈ q_mains
    theta_hat = q_mains / p if p > 0 else 0.0       # θ ≈ q/p
    delta_theta = 0.05

    L_idx = np.where(Pi_hat <= theta_hat + delta_theta)[0]

    # --- Shape checks for bounds (on low-probability variables L_theta) ---
    unimodal_tilde = None
    rconc_tilde_m12 = None
    rconc_hat_m14 = None
    bound_regime = "worst_case"

    if L_idx.size >= 3:
        support_tilde_L, pmf_tilde_L = pmf_from_probs(Pi_tilde[L_idx], B_eff=K_eff)
        support_hat_L, pmf_hat_L = pmf_from_probs(Pi_hat[L_idx], B_eff=2 * K_eff)

        if pmf_tilde_L.size > 0:
            unimodal_tilde = is_unimodal(pmf_tilde_L)
            rconc_tilde_m12 = r_concave_ok(pmf_tilde_L, r=-0.5)
        if pmf_hat_L.size > 0:
            rconc_hat_m14 = r_concave_ok(pmf_hat_L, r=-0.25)

        bound_regime = "worst_case"
        if unimodal_tilde is True:
            bound_regime = "unimodal"
        if rconc_tilde_m12 is True and rconc_hat_m14 is True:
            bound_regime = "r_concave"
    else:
        bound_regime = "worst_case (few L_theta vars)"
        # Plot Pi_tilde pmf for L_theta variables to check unimodality
    out_png = os.path.join(CONFIG["RESULTS_DIR"], f"pmf_Pi_tilde_Ltheta_.png")
    plot_pmf(
        support_tilde_L,
        pmf_tilde_L,
        title=f"Empirical pmf of $\\tilde{{\\Pi}}_B$ on $L_\\theta$)",
        out_png=out_png,
        uni_ok=is_unimodal(pmf_tilde_L),
        r_ok=r_concave_ok(pmf_tilde_L, r=-0.5),
        r_val="-1/2"
    )
    print(f"Saved Pi_tilde pmf plot for L_theta → {out_png}")
    return {
        "mains_freq": mains_freq,
        "inter_freq": inter_freq,
        "K_mains_halves": K_mains_halves,
        "K_inter_halves": K_inter_halves,
        "q_mains": q_mains,
        "q_inter": q_inter,
        "q_total": q_total,
        "q_mains_sum": q_mains_sum,
        "Pi_hat": Pi_hat,
        "Pi_tilde": Pi_tilde,
        "theta_hat": theta_hat,
        "L_idx": L_idx,
        "unimodal_tilde": unimodal_tilde,
        "rconc_tilde_m12": rconc_tilde_m12,
        "rconc_hat_m14": rconc_hat_m14,
        "bound_regime": bound_regime,
        "K_eff": K_eff,
    }

def main():
    ensure_dir(CONFIG["RESULTS_DIR"])
    tau_list = [0.5, 0.55, 0.6, 0.65, 0.7]
    data_files = [
        ("C:/Users/enthe/Desktop/Thesis/data/tgs_data/tgs_dataset_normalized.csv", "imbalanced"),
        ("C:/Users/enthe/Desktop/Thesis/data/tgs_data/tgs_dataset_normalized_balanced.csv", "balanced"),
    ]

    for data_path, balance_tag in data_files:
        X, y, x_cols = read_tgs_csv(data_path)
        p = X.shape[1]
        print(f"Running CPSS with mains + all pairwise interactions for {balance_tag} (p={p})")

        res = cpss_stability_selection_tgs(
            X, y,
            B_pairs=CONFIG["B"],
            Cs=CONFIG["logreg_cs"],
            seed=CONFIG["seed"]
        )
        print(f"q_mains: {res['q_mains']:.2f}, q_inters: {res['q_inter']:.2f}, q_total: {res['q_total']:.2f}")
                # Plot empirical pmf for mains
        pmf_mains = empirical_pmf(res["K_mains_halves"])
        out_png_mains = os.path.join(CONFIG["RESULTS_DIR"], f"pmf_mains_{balance_tag}.png")
        plot_pmf(
            pmf_mains["support"],
            pmf_mains["pmf"],
            title=f"Empirical pmf of mains (selected per half) for {balance_tag}",
            out_png=out_png_mains,
            uni_ok=is_unimodal(pmf_mains["pmf"]),
            r_ok=r_concave_ok(pmf_mains["pmf"], r=-0.5),
            r_val="-1/2"
        )
        print(f"Saved mains pmf plot → {out_png_mains}")

        # Plot empirical pmf for interactions
        pmf_inters = empirical_pmf(res["K_inter_halves"])
        out_png_inters = os.path.join(CONFIG["RESULTS_DIR"], f"pmf_inters_{balance_tag}.png")
        plot_pmf(
            pmf_inters["support"],
            pmf_inters["pmf"],
            title=f"Empirical pmf of interactions (selected per half) for {balance_tag}",
            out_png=out_png_inters,
            uni_ok=is_unimodal(pmf_inters["pmf"]),
            r_ok=r_concave_ok(pmf_inters["pmf"], r=-0.5),
            r_val="-1/2"
        )
        print(f"Saved interactions pmf plot → {out_png_inters}")

        # Plot empirical pmf for total (mains + interactions)
        K_total_halves = [a + b for a, b in zip(res["K_mains_halves"], res["K_inter_halves"])]
        pmf_total = empirical_pmf(K_total_halves)
        out_png_total = os.path.join(CONFIG["RESULTS_DIR"], f"pmf_total_{balance_tag}.png")
        plot_pmf(
            pmf_total["support"],
            pmf_total["pmf"],
            title=f"Empirical pmf of total (selected per half) for {balance_tag}",
            out_png=out_png_total,
            uni_ok=is_unimodal(pmf_total["pmf"]),
            r_ok=r_concave_ok(pmf_total["pmf"], r=-0.5),
            r_val="-1/2"
        )
        print(f"Saved total pmf plot → {out_png_total}")
        mains_freq = res["mains_freq"]
        inter_freq = res["inter_freq"]

        # --- NEW: print q and bound regime ---
        print(f"Estimated q_mains (E|S_n/2|, mains only) for {balance_tag}: {res['q_mains']:.3f}")
        print(f"Estimated q_total (E|S_n/2|, mains + interactions) for {balance_tag}: {res['q_total']:.3f}")
        print(f"Check: sum_k Pi_hat(k) = {res['q_mains_sum']:.3f} (should be ~ q_mains = {res['q_mains']:.3f})")
        print(f"theta_hat = q_mains / p = {res['theta_hat']:.4f}")
        print(f"CPSS bound regime suggested by shape checks: {res['bound_regime']}")
        print(f"  Unimodal Π̃_B on L_theta: {res['unimodal_tilde']}")
        print(f"  r-concave Π̃_B (r=-1/2) on L_theta: {res['rconc_tilde_m12']}")
        print(f"  r-concave Π̂_B (r=-1/4) on L_theta: {res['rconc_hat_m14']}")

        # Optional: print worst-case bound for each tau using q_mains
        for tau in tau_list:
            bound_val = conservative_false_selection_bound(res["q_mains"], tau, p)
            print(f"  Worst-case bound E[V] ≤ {bound_val:.3f} for tau={tau}")

        # Save results
        mains_freq_path = os.path.join(CONFIG["RESULTS_DIR"], f"stability_mains_tgs_l1logistic_{balance_tag}.csv")
        inter_freq_path = os.path.join(CONFIG["RESULTS_DIR"], f"stability_interactions_tgs_l1logistic_{balance_tag}.csv")
        pd.DataFrame({"freq": mains_freq}, index=x_cols).to_csv(mains_freq_path)

        rows = []
        for i in range(p):
            for j in range(i, p):
                rows.append({"i": i, "j": j, "freq": float(inter_freq[i, j])})
        pd.DataFrame(rows).to_csv(inter_freq_path, index=False)

        print(f"Saved mains CPSS frequencies → {mains_freq_path}")
        print(f"Saved interaction CPSS frequencies → {inter_freq_path}")

        # Save selected mains/interactions for each tau threshold
        mains_freq_df = pd.read_csv(mains_freq_path, index_col=0)
        inter_freq_df = pd.read_csv(inter_freq_path)
        inter_freq_df = inter_freq_df[inter_freq_df["i"] < inter_freq_df["j"]]

        for freq_threshold in tau_list:
            selected_mains = list(mains_freq_df[mains_freq_df["freq"] > freq_threshold].index)
            selected_interactions = [
                [x_cols[int(row["i"])], x_cols[int(row["j"])]]
                for _, row in inter_freq_df.iterrows()
                if row["freq"] > freq_threshold
            ]
            thresh_json = {
                "selected_mains": selected_mains,
                "selected_interactions": selected_interactions
            }
            thresh_json_path = os.path.join(
                CONFIG["RESULTS_DIR"],
                f"stability_selected_tgs_l1logistic_thresh_{freq_threshold}_{balance_tag}.json"
            )
            with open(thresh_json_path, "w") as f:
                json.dump(thresh_json, f, indent=2)
            print(f"Saved mains/interactions above threshold JSON → {thresh_json_path}")

        # Save top 10 mains/interactions as JSON (once per data file)
        top_n = 10
        top_mains = mains_freq_df["freq"].sort_values(ascending=False).head(top_n)
        selected_mains = list(top_mains.index)
        top_inter = inter_freq_df.sort_values("freq", ascending=False).head(top_n)
        selected_interactions = [
            [x_cols[int(row["i"])], x_cols[int(row["j"])]]
            for _, row in top_inter.iterrows()
        ]
        top_json = {
            "selected_mains": selected_mains,
            "selected_interactions": selected_interactions
        }
        top_json_path = os.path.join(CONFIG["RESULTS_DIR"], f"top10_stability_selected_tgs_l1logistic_{balance_tag}.json")
        with open(top_json_path, "w") as f:
            json.dump(top_json, f, indent=2)
        print(f"Saved top 10 mains/interactions JSON → {top_json_path}")

if __name__ == "__main__":
    main()
