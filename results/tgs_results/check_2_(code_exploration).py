"""
Pooled feature-lag autocorrelation across miRNA features,
and AR(1) geometric-decay check.

- Loads a CSV, optionally drops a label column.
- Centers each feature (column-wise).
- Computes r_hat_k = avg_cov_lag_k / avg_var   (covariance-based)
- Also computes avg of adjacent *correlations* (variance-robust).
- Estimates rho_hat from lag-1 and compares r_hat_k to rho_hat**k.
- Optional bootstrap CIs across feature pairs.


"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------- Config --------------------
CSV_PATH  = r"C:/Users/enthe/Desktop/Thesis/data/tgs_data/tgs_dataset_normalized.csv"
#CSV_PATH  = r"C:\Users\enthe\Desktop\Thesis\data\simulated_data\simulated_dataset_500_0.05_20_0.0.csv"
LABEL_COL = "MCI"       # set to None if no label column
MAX_LAG   = 10          # maximum feature lag (k) to evaluate
DO_BOOT   = True        # set False to skip bootstrapping
N_BOOT    = 1000        # bootstrap resamples
RANDOM_SEED = 42
SAVE_FIG  = False
FIG_PATH  = "pooled_feature_acf.png"
# ------------------------------------------------



def load_matrix(csv_path, label_col=None):
    df = pd.read_csv(csv_path)
    if label_col is not None and label_col in df.columns:
        df = df.drop(columns=[label_col])
    X = df.to_numpy(dtype=float)
    return X

def center_columns(X):
    return X - X.mean(axis=0, keepdims=True)

def pooled_acf_cov(Xc, max_lag):
    """
    Covariance-based pooled ACF across features:
    r_hat_k = avg_cov_lag_k / avg_variance
    """
    n, p = Xc.shape
    max_lag = min(max_lag, p - 1)
    # avg variance across columns (ddof=1)
    g0 = np.mean(np.var(Xc, axis=0, ddof=1))
    r = []
    n_pairs = []
    for k in range(1, max_lag + 1):
        covs = []
        for i in range(p - k):
            c = np.cov(Xc[:, i], Xc[:, i + k], ddof=1)[0, 1]
            covs.append(c)
        covs = np.asarray(covs, float)
        r.append(covs.mean() / g0 if covs.size else np.nan)
        n_pairs.append(covs.size)
    return np.array(r), np.array(n_pairs), g0

def pooled_acf_corr(Xc, max_lag):
    """
    Correlation-based pooled ACF across features:
    avg of pairwise correlations at each lag (more robust when feature variances differ).
    """
    n, p = Xc.shape
    max_lag = min(max_lag, p - 1)
    s = Xc.std(axis=0, ddof=1)
    r = []
    n_pairs = []
    for k in range(1, max_lag + 1):
        corrs = []
        for i in range(p - k):
            si, sj = s[i], s[i + k]
            if si == 0 or sj == 0:
                continue  # skip constant features
            cov_ij = np.cov(Xc[:, i], Xc[:, i + k], ddof=1)[0, 1]
            corrs.append(cov_ij / (si * sj))
        corrs = np.asarray(corrs, float)
        r.append(corrs.mean() if corrs.size else np.nan)
        n_pairs.append(corrs.size)
    return np.array(r), np.array(n_pairs)

def bootstrap_ci_across_pairs(Xc, max_lag, corr_based=True, n_boot=1000, seed=0):
    """
    Bootstrap 95% CIs by resampling feature pairs with replacement at each lag.
    corr_based=True -> resample correlations; else covariances/variance ratio.
    """
    rng = np.random.default_rng(seed)
    n, p = Xc.shape
    max_lag = min(max_lag, p - 1)
    # Precompute per-lag vectors of pairwise stats
    perlag_vals = []
    if corr_based:
        s = Xc.std(axis=0, ddof=1)
    else:
        g0 = np.mean(np.var(Xc, axis=0, ddof=1))
    for k in range(1, max_lag + 1):
        vals = []
        for i in range(p - k):
            if corr_based:
                si, sj = s[i], s[i + k]
                if si == 0 or sj == 0:
                    continue
                cov_ij = np.cov(Xc[:, i], Xc[:, i + k], ddof=1)[0, 1]
                vals.append(cov_ij / (si * sj))
            else:
                cov_ij = np.cov(Xc[:, i], Xc[:, i + k], ddof=1)[0, 1]
                vals.append(cov_ij / g0)
        perlag_vals.append(np.asarray(vals, float))

    # Bootstrap
    q_lo, q_hi = [], []
    for vals in perlag_vals:
        if vals.size == 0:
            q_lo.append(np.nan); q_hi.append(np.nan); continue
        boot_means = []
        m = vals.size
        for _ in range(n_boot):
            idx = rng.integers(0, m, size=m)
            boot_means.append(vals[idx].mean())
        qs = np.quantile(boot_means, [0.025, 0.975])
        q_lo.append(qs[0]); q_hi.append(qs[1])
    return np.array(q_lo), np.array(q_hi)

def main():

    X = load_matrix(CSV_PATH, LABEL_COL)
    if X.shape[1] < 2:
        raise ValueError("Need at least 2 features (columns) to compute lag-1.")
    Xc = center_columns(X)
    # --- Greedy feature reordering to maximize adjacent correlations ---
    p = Xc.shape[1]
    corr = np.corrcoef(Xc, rowvar=False)
    np.fill_diagonal(corr, 0)  # Ignore self-correlation

    used = set()
    order = [0]  # Start from first feature
    used.add(0)
    for _ in range(1, p):
        last = order[-1]
        next_idx = np.argmax(corr[last, :])
        while next_idx in used:
            corr[last, next_idx] = -np.inf
            next_idx = np.argmax(corr[last, :])
        order.append(next_idx)
        used.add(next_idx)

    # Rearranged Xc
    Xc_reordered = Xc[:, order]    
    

    # Covariance-based pooled ACF
    r_cov, n_pairs_cov, g0 = pooled_acf_cov(Xc, MAX_LAG)
    # Correlation-based pooled ACF (recommended)
    r_cor, n_pairs_cor = pooled_acf_corr(Xc, MAX_LAG)

        # Covariance-based pooled ACF
    r_cov, n_pairs_cov, g0 = pooled_acf_cov(Xc_reordered, MAX_LAG)
    # Correlation-based pooled ACF (recommended)
    r_cor, n_pairs_cor = pooled_acf_corr(Xc_reordered, MAX_LAG)
    # ...rest of code unchanged...

    # AR(1) estimate and geometric decay
    rho_hat_cov = r_cov[0] if np.isfinite(r_cov[0]) else np.nan
    rho_hat_cor = r_cor[0] if np.isfinite(r_cor[0]) else np.nan
    ks = np.arange(1, len(r_cor) + 1)
    geom_cov = rho_hat_cov ** ks if np.isfinite(rho_hat_cov) else np.full_like(ks, np.nan, float)
    geom_cor = rho_hat_cor ** ks if np.isfinite(rho_hat_cor) else np.full_like(ks, np.nan, float)

    # Optional bootstrap CIs
    if DO_BOOT:
        qlo_cor, qhi_cor = bootstrap_ci_across_pairs(Xc, MAX_LAG, corr_based=True, n_boot=N_BOOT, seed=RANDOM_SEED)
        qlo_cov, qhi_cov = bootstrap_ci_across_pairs(Xc, MAX_LAG, corr_based=False, n_boot=N_BOOT, seed=RANDOM_SEED)
    else:
        qlo_cor = qhi_cor = qlo_cov = qhi_cov = np.full_like(ks, np.nan, float)

    # ---- Reporting ----
    print(f"n samples = {X.shape[0]}, p features = {X.shape[1]}")
    print(f"Pairs per lag (corr-based): {n_pairs_cor.tolist()}")
    print(f"rho_hat (corr-based, lag-1 mean corr): {rho_hat_cor:.4f}")
    print(f"rho_hat (cov-based, g1/g0):            {rho_hat_cov:.4f}")

    # ---- Plot (corr-based, recommended) ----
    plt.figure(figsize=(7.5, 5.0))
    plt.plot(ks, r_cor, marker='o', label=r"Empirical $\hat r_k$ (corr-based)")
    if DO_BOOT and np.isfinite(qlo_cor).any():
        plt.fill_between(ks, qlo_cor, qhi_cor, alpha=0.2, label="Bootstrap 95% CI")
    if np.isfinite(rho_hat_cor):
        plt.plot(ks, geom_cor, linestyle="--", label=r"$\hat\rho^{\,k}$ (AR(1))")
    plt.axhline(0, linewidth=0.8)
    plt.xlabel("Feature lag k")
    plt.ylabel(r"Pooled correlation $\hat r_k$")
    plt.title("Pooled feature-lag autocorrelation vs. AR(1) geometric decay")
    plt.legend()
    plt.tight_layout()
    if SAVE_FIG:
        plt.savefig(FIG_PATH, dpi=150)
        print(f"Saved figure to: {FIG_PATH}")
    plt.show()

    # ---- (Optional) Second plot: covariance-based version ----
    plt.figure(figsize=(7.5, 5.0))
    plt.plot(ks, r_cov, marker='o', label=r"Empirical $\hat r_k$ (cov-based)")
    if DO_BOOT and np.isfinite(qlo_cov).any():
        plt.fill_between(ks, qlo_cov, qhi_cov, alpha=0.2, label="Bootstrap 95% CI")
    if np.isfinite(rho_hat_cov):
        plt.plot(ks, geom_cov, linestyle="--", label=r"$\hat\rho^{\,k}$ (AR(1))")
    plt.axhline(0, linewidth=0.8)
    plt.xlabel("Feature lag k")
    plt.ylabel(r"Pooled cov/var ratio $\hat r_k$")
    plt.title("Pooled feature-lag autocovariance ratio vs. AR(1) decay")
    plt.legend()
    plt.tight_layout()
    if SAVE_FIG:
        base = FIG_PATH.rsplit(".", 1)[0]
        path2 = f"{base}_cov.png"
        plt.savefig(path2, dpi=150)
        print(f"Saved figure to: {path2}")
    plt.show()

if __name__ == "__main__":
    main()
