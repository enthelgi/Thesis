import os
import json
import numpy as np
import pandas as pd
from plasso import PliableLasso
from sklearn.linear_model import LassoCV
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
    "lasso_alphas": np.linspace(0.0001, 0.05, 30),
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

def select_z_lasso(X, y, x_cols, max_k=10, alphas=None, seed=13):
    lasso = LassoCV(alphas=alphas, cv=5, random_state=seed)
    lasso.fit(X, y)
    coefs = np.abs(lasso.coef_)
    main_indices = np.where(coefs > 1e-8)[0]
    if len(main_indices) > max_k:
        sorted_idx = main_indices[np.argsort(-coefs[main_indices])]
        main_indices = sorted_idx[:max_k]
    return main_indices

def empirical_pmf(counts):
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
    if p.sum() <= 0: return False
    idx_mode = int(np.argmax(p))
    left = np.all(np.diff(p[:idx_mode+1]) >= -tol)
    right = np.all(np.diff(p[idx_mode:]) <= tol)
    return bool(left and right)

def r_concave_ok(pmf, r=-0.5, tol=1e-10):
    p = np.asarray(pmf, float)
    mask = p > 0
    if mask.sum() < 3:
        return True
    x = p[mask] ** r
    second = x[:-2] - 2 * x[1:-1] + x[2:]
    return bool(np.all(second <= tol))

def conservative_false_selection_bound(q, tau, p):
    if tau <= 0.5:
        return float("inf")
    return q**2 / (p * max(1e-12, (2.0 * tau - 1.0)))

def plot_pmf(support, pmf, title, out_png, uni_ok, r_ok, r_val):
    plt.figure(figsize=(8,5))
    ax = sns.barplot(x=[str(k) for k in support], y=pmf)
    ax.set_title(f"{title}\nUnimodal={uni_ok} | r-concave(r={r_val})={r_ok}")
    ax.set_xlabel("K (selected per half)"); ax.set_ylabel("Empirical pmf")
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()

def cpss_stability_selection_tgs(X, y, z_idx, B_pairs=50, cv=0.1, seed=123):
    rng = np.random.default_rng(seed)
    n, p = X.shape
    Z_full = X[:, z_idx]
    p_mod = Z_full.shape[1]

    mains_pair_count = np.zeros(p, dtype=float)
    inter_pair_count = np.zeros((p, p), dtype=float)
    K_eff = 0
    K_mains_halves = []
    K_inter_halves = []

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
        rng.shuffle(A); rng.shuffle(B)

        if np.unique(y[A]).size < 2 or np.unique(y[B]).size < 2:
            continue

        def fit_and_select(sub_idx):
            Xb, yb = X[sub_idx], y[sub_idx]
            Zb = Z_full[sub_idx]
            mdl = PliableLasso(cv=cv, max_interaction_terms= 1000, verbose=False, eps=1e-4, normalize=True)
            mdl.fit(Xb, Zb, yb)
            beta = getattr(mdl, "beta_", getattr(mdl, "beta", None))
            theta = getattr(mdl, "theta_", getattr(mdl, "theta", None))
            mains_sel = np.zeros(p, dtype=bool)
            inter_sel = np.zeros((p, p), dtype=bool)
            if beta is not None:
                mains_sel = (np.abs(beta) > 1e-8)
            if theta is not None:
                sel = (np.abs(theta) > 1e-8)
                for i in range(p):
                    for jj in range(p_mod):
                        if sel[i, jj]:
                            j = int(z_idx[jj])
                            inter_sel[i, j] = True
                            inter_sel[j, i] = True
            return mains_sel, inter_sel

        try:
            # A -> .txt
            # subprocess (Rscript [diavase .txt]) -> tha ftiaxksi dyo alla .txt : main_sel.txt , inter_sel.txt
            # 
            mains_A, inter_A = fit_and_select(A)
            mains_B, inter_B = fit_and_select(B)
            K_mains_halves.append(int(np.sum(mains_A)))
            K_inter_halves.append(int(np.sum(inter_A)))
            K_mains_halves.append(int(np.sum(mains_B)))
            K_inter_halves.append(int(np.sum(inter_B)))
        except Exception:
            continue

        if mains_A is None or mains_B is None or inter_A is None or inter_B is None:
            continue

        mains_pair = np.logical_or(mains_A, mains_B)
        inter_pair = np.logical_or(inter_A, inter_B)
        mains_pair_count += mains_pair.astype(float)
        inter_pair_count += inter_pair.astype(float)
        K_eff += 1

    if K_eff == 0:
        raise RuntimeError("All CPSS pairs failed; try adjusting parameters or random seed.")

    mains_freq = mains_pair_count / K_eff
    inter_freq = inter_pair_count / K_eff
    return mains_freq, inter_freq, K_mains_halves, K_inter_halves

def main():
    ensure_dir(CONFIG["RESULTS_DIR"])
    tau_list = [0.5, 0.55, 0.6, 0.65, 0.7]
    data_files = [
        ("C:/Users/enthe/Desktop/Thesis/data/tgs_data/tgs_dataset_normalized.csv", "imbalanced"),
        ("C:/Users/enthe/Desktop/Thesis/data/tgs_data/tgs_dataset_normalized_balanced.csv", "balanced"),
    ]

    for data_path, balance_tag in data_files:
        X, y, x_cols = read_tgs_csv(data_path)
        z_idx = select_z_lasso(X, y, x_cols, max_k=CONFIG["mod_k"], alphas=CONFIG["lasso_alphas"], seed=CONFIG["seed"])
        print(f"Selected Z indices (Lasso) for {balance_tag}: {z_idx}")
        mains_freq, inter_freq, K_mains_halves, K_inter_halves = cpss_stability_selection_tgs(
            X, y, z_idx,
            B_pairs=CONFIG["B"],
            cv=CONFIG["cv"],
            seed=CONFIG["seed"]
        )
        # Save results
        mains_freq_path = os.path.join(CONFIG["RESULTS_DIR"], f"stability_mains_tgs_{balance_tag}.csv")
        inter_freq_path = os.path.join(CONFIG["RESULTS_DIR"], f"stability_interactions_tgs_{balance_tag}.csv")
        pd.DataFrame({"freq": mains_freq}, index=x_cols).to_csv(mains_freq_path)
        rows = []
        p = X.shape[1]
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
                f"stability_selected_tgs_thresh_{freq_threshold}_{balance_tag}.json"
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
        top_json_path = os.path.join(CONFIG["RESULTS_DIR"], f"top10_stability_selected_tgs_{balance_tag}.json")
        with open(top_json_path, "w") as f:
            json.dump(top_json, f, indent=2)
        print(f"Saved top 10 mains/interactions JSON → {top_json_path}")

if __name__ == "__main__":
    main()