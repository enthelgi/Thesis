"""
Generates simulated datasets with a continuous response across a grid and saves
ONE CSV per combination:
  simulated_dataset_<n>_<noise_sd>_<p>_<rho>.csv

CSV columns:
  X1, X2, ..., Xp, y, mu

where:
  - y  = mu + epsilon
  - mu = alpha + X * beta + (X_j * X_k) * gamma_jk
  - epsilon ~ N(0, noise_sd^2)

Also saves a truth JSON per dataset:
  truth_simulated_dataset_<n>_<noise_sd>_<p>_<rho>.json
with:
  - n, p, rho, alpha, noise_sd
  - main_idx, interaction_core_idx, interaction_pairs
  - beta (main effect coefficients)
  - gamma (interaction coefficients)

A manifest 'simulated_dataset_manifest.csv' lists all datasets.
"""

import os
from itertools import product
import numpy as np
import pandas as pd
import json

# ---------------- Helpers ----------------

def make_ar1_cov(p, rho):
    idx = np.arange(p)
    return rho ** np.abs(idx[:, None] - idx[None, :])

def sample_gaussian_X(n, p, rho, rng):
    """
    Sample X ~ N(0, Sigma) with AR(1)-type correlation structure.
    """
    if rho == 0.0:
        return rng.standard_normal((n, p))
    Sigma = make_ar1_cov(p, rho)
    # small ridge for numerical stability
    L = np.linalg.cholesky(Sigma + 1e-12 * np.eye(p))
    Z = rng.standard_normal((n, p))
    return Z @ L.T

def choose_truth_indices(p, num_main=10, num_interaction_core=5, rng=None):
    """
    Choose indices for:
      - main effects
      - a core subset of those for interactions
      - all pairwise interactions within that core
    """
    if rng is None:
        rng = np.random.default_rng()
    mains = rng.choice(p, size=num_main, replace=False)
    core  = rng.choice(mains, size=num_interaction_core, replace=False)
    pairs = []
    for i_idx, i in enumerate(core):
        for j in core[i_idx + 1:]:
            pairs.append((int(i), int(j)))
    return list(map(int, mains)), list(map(int, core)), pairs

def build_coefficients(main_idx, interaction_pairs, beta_scale, gamma_scale, rng):
    """
    Build dictionaries of coefficients:
      beta[j] for main effect j
      gamma["j_k"] for interaction between j and k
    """
    beta = {int(j): float(beta_scale * rng.normal()) for j in main_idx}
    gamma = {f"{int(j)}_{int(k)}": float(gamma_scale * rng.normal())
             for (j, k) in interaction_pairs}
    return beta, gamma

def simulate_once(
    n,
    p,
    rho,
    noise_sd,
    rng,
    num_main=10,
    num_interaction_core=5,
    beta_scale=0.6,
    gamma_scale=0.5,
):
    """
    Simulate one dataset with continuous outcome:

      y = alpha + X * beta + sum_{(j,k)} gamma_{jk} X_j X_k + eps
      eps ~ N(0, noise_sd^2)

    Returns:
      X  : (n, p)
      y  : (n,)
      mu : (n,)  (true mean signal without noise)
      truth : dict with all ground-truth info
    """
    # Design matrix
    X = sample_gaussian_X(n, p, rho, rng)

    # Truth structure
    mains, cores, pairs = choose_truth_indices(
        p,
        num_main=num_main,
        num_interaction_core=num_interaction_core,
        rng=rng,
    )
    beta, gamma = build_coefficients(
        mains,
        pairs,
        beta_scale=beta_scale,
        gamma_scale=gamma_scale,
        rng=rng,
    )

    # Linear predictor (signal)
    mu = np.zeros(n)

    # Main effects
    if beta:
        idx = np.array(list(beta.keys()), dtype=int)
        mu += X[:, idx] @ np.array([beta[j] for j in idx])

    # Interaction effects
    for key, g in gamma.items():
        j, k = map(int, key.split("_"))
        mu += g * (X[:, j] * X[:, k])

    # Intercept (can be 0 or set to any constant if you like)
    alpha = 0.0
    mu = alpha + mu

    # Gaussian noise
    eps = rng.normal(loc=0.0, scale=noise_sd, size=n)
    y = mu + eps

    truth = {
        "n": int(n),
        "p": int(p),
        "rho": float(rho),
        "alpha": float(alpha),
        "noise_sd": float(noise_sd),
        "main_idx": mains,
        "interaction_core_idx": cores,
        "interaction_pairs": pairs,
        "beta": beta,
        "gamma": gamma,
    }
    return X, y, mu, truth

# ---------------- Config ----------------

base_n = 500
base_p = 20
seed = 42

# n and p grids as before
n_multipliers = (0.5, 1.0, 2.0)
p_multipliers = (1, 25)

# Now this is a grid over noise standard deviations
noise_sds = (0.5, 1.0, 2.0)

# Correlation parameters
rhos = (0.0, 0.5, 0.8)

# Truth structure / effect sizes
num_main = 10
num_interaction_core = 5
beta_scale = 0.6
gamma_scale = 0.5

out_dir = "C:/Users/enthe/Desktop/Thesis/data/simulated_data_continuous"
os.makedirs(out_dir, exist_ok=True)

# ---------------- Main ----------------

def main():
    rng = np.random.default_rng(seed)
    manifest_rows = []

    for m_n, noise_sd, m_p, rho in product(
        n_multipliers, noise_sds, p_multipliers, rhos
    ):
        n = int(round(base_n * m_n))
        p = int(base_p * m_p)
        if p < num_main:
            continue

        X, y, mu, truth = simulate_once(
            n,
            p,
            rho,
            noise_sd,
            rng,
            num_main=num_main,
            num_interaction_core=num_interaction_core,
            beta_scale=beta_scale,
            gamma_scale=gamma_scale,
        )

        # Save CSV
        fname = f"simulated_dataset_{n}_{noise_sd}_{p}_{rho}.csv"
        csv_path = os.path.join(out_dir, fname)
        cols = [f"X{j+1}" for j in range(p)]
        df = pd.DataFrame(X, columns=cols)
        df["y"] = y
        df["mu"] = mu  # true signal (noiseless)
        df.to_csv(csv_path, index=False)

        # Save truth JSON
        truth_fname = f"truth_{os.path.splitext(fname)[0]}.json"
        truth_path = os.path.join(out_dir, truth_fname)
        with open(truth_path, "w") as f:
            json.dump(truth, f)

        manifest_rows.append({
            "filename": fname,
            "n": int(n),
            "p": int(p),
            "noise_sd": float(noise_sd),
            "rho": float(rho),
            "y_mean": float(y.mean()),
            "y_sd": float(y.std(ddof=1)),
            "num_mains": len(truth["main_idx"]),
            "num_interactions": len(truth["interaction_pairs"]),
        })
        print("Saved", csv_path)

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(os.path.join(out_dir, "simulated_dataset_manifest.csv"), index=False)
    print("All done. Created", len(manifest_rows), "datasets.")

if __name__ == "__main__":
    main()
