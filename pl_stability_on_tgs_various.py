import os
import json
import numpy as np
import pandas as pd
from plasso import PliableLasso

# --- CONFIG ---
CONFIG = {
    #"DATA_PATH": "C:/Users/enthe/Desktop/Thesis/data/tgs_data/tgs_dataset_normalized.csv",
    "DATA_PATH": "C:/Users/enthe/Desktop/Thesis/data/tgs_data/tgs_dataset_normalized_balanced.csv",
    "RESULTS_DIR": "C:/Users/enthe/Desktop/Thesis/results/tgs_results",
    "FIT_JSON": "selected_tgs_imbalanced.json",  # or "selected_tgs_balanced.json"
    "mod_k": 10,
    "B": 50,
    "cv": 0.05,
    "max_interaction_terms": 200,
    "seed": 42,
}

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def read_tgs_csv(path):
    df = pd.read_csv(path)
    x_cols = [c for c in df.columns if c != "MCI"]
    X = df[x_cols].to_numpy()
    y = df["MCI"].to_numpy().ravel()
    return X, y, x_cols

def select_z_from_fit_json(json_path, x_cols, max_k=10):
    with open(json_path) as f:
        fit_summary = json.load(f)
    selected_mains_names = fit_summary.get("selected_mains", [])
    print(f"[DEBUG] selected_mains_names from fit JSON: {selected_mains_names}")
    z_idx = [x_cols.index(name) for name in selected_mains_names if name in x_cols]
    print(f"[DEBUG] z_idx after mapping to x_cols: {z_idx}")
    if len(z_idx) > max_k:
        z_idx = z_idx[:max_k]
        print(f"[DEBUG] z_idx truncated to max_k={max_k}: {z_idx}")
    return z_idx

def cpss_stability_selection_tgs(X, y, z_idx, B_pairs=100, cv=0.2, max_interaction_terms=50, seed=42):
    np.random.seed(seed)
    p = X.shape[1]
    mains_freq = np.zeros(p)
    inter_freq = np.zeros((p, p))
    n = X.shape[0]
    print(f"[DEBUG] Starting CPSS with z_idx: {z_idx}")
    for b in range(B_pairs):
        idx = np.random.choice(n, n // 2, replace=False)
        Xb, yb = X[idx], y[idx]
        Zb = Xb[:, z_idx]
        print(f"[DEBUG] Subsample {b+1}/{B_pairs}: Zb shape = {Zb.shape}")
        model = PliableLasso(alpha=0.05, cv=cv, max_interaction_terms=max_interaction_terms, normalize=True, verbose=False)
        try:
            model.fit(Xb, Zb, yb)
            beta = getattr(model, "beta_", None)
            if beta is None: beta = getattr(model, "coef_", None)
            theta = getattr(model, "theta_", None)
            print(f"[DEBUG] Subsample {b+1}: Nonzero mains = {np.sum(np.abs(beta) > 1e-8) if beta is not None else 'N/A'}")
            print(f"[DEBUG] Subsample {b+1}: Nonzero interactions = {np.sum(np.abs(theta) > 1e-8) if theta is not None else 'N/A'}")
            if beta is not None:
                for j in range(min(p, len(beta))):
                    if abs(beta[j]) > 1e-8:
                        mains_freq[j] += 1
            if theta is not None:
                T = np.asarray(theta, float)
                for i in range(p):
                    for jj in range(T.shape[1]):
                        if abs(T[i, jj]) > 1e-8:
                            inter_freq[i, z_idx[jj]] += 1
        except Exception as e:
            print(f"[ERROR] Subsample {b+1}: {e}")
            continue
    mains_freq /= B_pairs
    inter_freq /= B_pairs
    print(f"[DEBUG] Final mains_freq: {mains_freq}")
    print(f"[DEBUG] Final inter_freq (sum): {np.sum(inter_freq)}")
    return mains_freq, inter_freq

def main():
    ensure_dir(CONFIG["RESULTS_DIR"])
    X, y, x_cols = read_tgs_csv(CONFIG["DATA_PATH"])
    fit_json_path = os.path.join(CONFIG["RESULTS_DIR"], CONFIG["FIT_JSON"])
    z_idx = select_z_from_fit_json(fit_json_path, x_cols, max_k=CONFIG["mod_k"])
    print(f"Selected Z indices (from fit JSON): {z_idx}")

    mains_freq, inter_freq = cpss_stability_selection_tgs(
        X, y, z_idx,
        B_pairs=CONFIG["B"],
        cv=CONFIG["cv"],
        max_interaction_terms=CONFIG["max_interaction_terms"],
        seed=CONFIG["seed"]
    )

    # Save mains frequencies
    mains_freq_path = os.path.join(CONFIG["RESULTS_DIR"], "stability_mains_tgs_various.csv")
    pd.DataFrame({"freq": mains_freq}, index=x_cols).to_csv(mains_freq_path)
    print(f"Saved mains CPSS frequencies → {mains_freq_path}")

    # Save interaction frequencies
    inter_freq_path = os.path.join(CONFIG["RESULTS_DIR"], "stability_interactions_tgs_various.csv")
    rows = []
    p = X.shape[1]
    for i in range(p):
        for j in range(i, p):
            rows.append({"i": i, "j": j, "freq": float(inter_freq[i, j])})
    pd.DataFrame(rows).to_csv(inter_freq_path, index=False)
    print(f"Saved interaction CPSS frequencies → {inter_freq_path}")

    # Save top 10 mains/interactions as JSON (by frequency, with names)
    top_n = 10
    mains_freq_df = pd.read_csv(mains_freq_path, index_col=0)
    top_mains = mains_freq_df["freq"].sort_values(ascending=False).head(top_n)
    selected_mains = list(top_mains.index)

    inter_freq_df = pd.read_csv(inter_freq_path)
    inter_freq_df = inter_freq_df[inter_freq_df["i"] < inter_freq_df["j"]]
    top_inter = inter_freq_df.sort_values("freq", ascending=False).head(top_n)
    selected_interactions = [
        [x_cols[int(row["i"])], x_cols[int(row["j"])]]
        for _, row in top_inter.iterrows()
    ]

    top_json = {
        "selected_mains": selected_mains,
        "selected_interactions": selected_interactions
    }
    top_json_path = os.path.join(CONFIG["RESULTS_DIR"], "top10_stability_selected_tgs_various.json")
    with open(top_json_path, "w") as f:
        json.dump(top_json, f, indent=2)
    print(f"Saved top 10 mains/interactions JSON → {top_json_path}")

    # Save mains/interactions above a threshold as JSON
    freq_threshold = 0.5
    selected_mains_thresh = list(mains_freq_df[mains_freq_df["freq"] > freq_threshold].index)
    selected_interactions_thresh = [
        [x_cols[int(row["i"])], x_cols[int(row["j"])]]
        for _, row in inter_freq_df.iterrows()
        if row["freq"] > freq_threshold
    ]
    thresh_json = {
        "selected_mains": selected_mains_thresh,
        "selected_interactions": selected_interactions_thresh
    }
    thresh_json_path = os.path.join(CONFIG["RESULTS_DIR"], f"stability_selected_tgs_various_thresh_{freq_threshold}.json")
    with open(thresh_json_path, "w") as f:
        json.dump(thresh_json, f, indent=2)
    print(f"Saved mains/interactions above threshold JSON → {thresh_json_path}")

if __name__ == "__main__":
    main()