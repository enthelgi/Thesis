import os
import numpy as np
import pandas as pd
from plasso import PliableLasso
from sklearn.linear_model import LassoCV

# --- CONFIG ---
CONFIG = {
    "DATASETS": [
        (r"C:\Users\enthe\Desktop\Thesis\data\tgs_data\tgs_dataset_normalized.csv", "imbalanced"),
        (r"C:\Users\enthe\Desktop\Thesis\data\tgs_data\tgs_dataset_normalized_balanced.csv", "balanced"),
    ],
    "B": 50,               # number of complementary pairs
    "cv": 0.1,
    "seed": 123,           # you can change/remove this; q should be similar anyway
    "mod_k": 10,
    "lasso_alphas": np.linspace(0.0001, 0.05, 30),
}

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

def cpss_stability_selection_tgs(X, y, z_idx, B_pairs=50, cv=0.1, seed=123):
    """
    Same as your function, but we'll just return the things we need for q.
    """
    rng = np.random.default_rng(seed)
    n, p = X.shape
    Z_full = X[:, z_idx]
    p_mod = Z_full.shape[1]

    mains_pair_count = np.zeros(p, dtype=float)
    inter_pair_count = np.zeros((p, p), dtype=float)
    K_eff = 0

    # we will collect per-half selected counts
    K_mains_halves = []
    K_inter_halves = []

    for _ in range(B_pairs):
        # stratified complementary halves
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

        # need at least 2 classes in each half
        if np.unique(y[A]).size < 2 or np.unique(y[B]).size < 2:
            continue

        def fit_and_select(sub_idx):
            Xb, yb = X[sub_idx], y[sub_idx]
            Zb = Z_full[sub_idx]
            mdl = PliableLasso(
                cv=cv,
                max_interaction_terms=1000,
                verbose=False,
                eps=1e-4,
                normalize=True,
            )
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
            mains_A, inter_A = fit_and_select(A)
            mains_B, inter_B = fit_and_select(B)

            # store per-half counts (this is what we need for q)
            K_mains_halves.append(int(np.sum(mains_A)))
            K_mains_halves.append(int(np.sum(mains_B)))
            K_inter_halves.append(int(np.sum(inter_A)))
            K_inter_halves.append(int(np.sum(inter_B)))
        except Exception:
            # skip this pair if model fails
            continue

        mains_pair = np.logical_or(mains_A, mains_B)
        inter_pair = np.logical_or(inter_A, inter_B)
        mains_pair_count += mains_pair.astype(float)
        inter_pair_count += inter_pair.astype(float)
        K_eff += 1

    if K_eff == 0:
        raise RuntimeError("All CPSS pairs failed; try adjusting parameters or random seed.")

    # you could also return mains_pair_count/inter_pair_count, but we don't need them for q
    return K_mains_halves, K_inter_halves

def main():
    for data_path, tag in CONFIG["DATASETS"]:
        print(f"\n=== {tag} dataset ===")
        X, y, x_cols = read_tgs_csv(data_path)
        z_idx = select_z_lasso(
            X, y, x_cols,
            max_k=CONFIG["mod_k"],
            alphas=CONFIG["lasso_alphas"],
            seed=CONFIG["seed"],
        )
        print(f"Z indices (modifier vars): {z_idx}")

        K_mains_halves, K_inter_halves = cpss_stability_selection_tgs(
            X, y, z_idx,
            B_pairs=CONFIG["B"],
            cv=CONFIG["cv"],
            seed=CONFIG["seed"],   # change/remove to see variability
        )

        K_mains_halves = np.asarray(K_mains_halves, dtype=float)
        K_inter_halves = np.asarray(K_inter_halves, dtype=float)

        q_mains = float(K_mains_halves.mean()) if K_mains_halves.size > 0 else 0.0
        q_inters = float(K_inter_halves.mean()) if K_inter_halves.size > 0 else 0.0

        print(f"q (mains, estimated)     = {q_mains:.4f}")
        print(f"q (interactions, rough)  = {q_inters:.4f}")
        print(f"#half-samples actually used (mains) = {K_mains_halves.size}")

if __name__ == "__main__":
    main()
