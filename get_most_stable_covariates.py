import os
import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support

N = 30  # Number of top stable mains/interactions to keep
THRESH = 1e-5  # Coefficient threshold for "chosen"
base_results_dir = "C:/Users/enthe/Desktop/Thesis/results"
fit_dir = os.path.join(base_results_dir, "fit_predcv")

sns.set_theme(style="whitegrid", context="talk")

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

def plot_overlap_bars(labels, counts, title, out_png):
    plt.figure(figsize=(9,7))
    ax = sns.barplot(x=labels, y=counts, hue=labels, palette="muted", legend=False)
    ax.set_title(title)
    ax.set_ylabel("Count")
    ax.set_xlabel("")
    ax.set_xticklabels(labels, rotation=20, ha="right")
    for i, v in enumerate(counts):
        ax.text(i, v + 0.2, str(v), ha="center", va="bottom", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()

def get_topN_stable(df, stable_indices, N, kind="mains"):
    if kind == "mains":
        stable_set = set(int(i) for i in stable_indices)
        df_stable = df[df.index.isin(stable_set)]
        df_sorted = df_stable.sort_values("freq", ascending=False)
        topN = df_sorted.head(N)
        return set(int(i) for i in topN.index)
    else:
        stable_set = set(tuple(sorted(pair)) for pair in stable_indices if len(pair) == 2)
        df["pair"] = df.apply(lambda row: tuple(sorted([row["i"], row["j"]])), axis=1)
        df_stable = df[df["pair"].isin(stable_set)]
        df_sorted = df_stable.sort_values("freq", ascending=False)
        topN = df_sorted.head(N)
        return set(tuple(sorted([row["i"], row["j"]])) for _, row in topN.iterrows())

def process_one(summary_path, mains_freq_csv, inter_freq_csv, out_dir, N):
    with open(summary_path) as f:
        summary = json.load(f)
    dataset = summary.get("dataset_stem", os.path.basename(summary_path))
    tau = summary.get("tau", None)

    # Load true indices from truth file if available
    truth_file = None
    folder = os.path.dirname(summary_path)
    candidate = os.path.join(folder, f"truth_{dataset}.json")
    if os.path.exists(candidate):
        truth_file = candidate
    else:
        candidate2 = os.path.join("C:/Users/enthe/Desktop/Thesis/data/simulated_data", f"truth_{dataset}.json")
        if os.path.exists(candidate2):
            truth_file = candidate2

    mains_true = set()
    inter_true = set()
    if truth_file:
        with open(truth_file) as f:
            truth = json.load(f)
        mains_true = set(truth.get("main_idx", []))
        inter_true = set(tuple(sorted(pair)) for pair in truth.get("interaction_pairs", []))

    # Load frequency CSVs
    mains_freq_df = pd.read_csv(mains_freq_csv)
    inter_freq_df = pd.read_csv(inter_freq_csv)

    # Get top N stable mains/interactions
    stable_mains = get_topN_stable(mains_freq_df, summary["stable_indices"]["mains"], N, kind="mains")
    stable_interactions = get_topN_stable(inter_freq_df, summary["stable_indices"]["interactions"], N, kind="interactions")

    # ---------------------------------------------------------------------
    # CHOSEN SECTION REMOVED (commented out)
    # best_model_path = os.path.join(fit_dir, f"best_model_{dataset}.json")
    # chosen_mains = set()
    # chosen_interactions = set()
    # if os.path.exists(best_model_path):
    #     with open(best_model_path, "r") as f:
    #         chosen_data = json.load(f)
    #         # Filter chosen mains by abs(coef) >= THRESH
    #         for idx, coef in chosen_data.get("main_coefficients", {}).items():
    #             if abs(coef) >= THRESH:
    #                 chosen_mains.add(int(idx))
    #         # Filter chosen interactions by abs(coef) >= THRESH
    #         for key, coef in chosen_data.get("interaction_coefficients", {}).items():
    #             i, j = map(int, key.split("_"))
    #             if abs(coef) >= THRESH:
    #                 chosen_interactions.add(tuple(sorted((i, j))))
    # ---------------------------------------------------------------------

    # Overlap counts (mains) — keep only stable, true, and stable ∩ true
    stable_and_true = stable_mains & mains_true

    mains_counts = {
        # "chosen_total": len(chosen_mains),
        "stable_total": len(stable_mains),
        "true_total": len(mains_true),
        # "chosen_and_stable": len(chosen_and_stable),
        # "chosen_and_true": len(chosen_and_true),
        "stable_and_true": len(stable_and_true),
        # "all_three": len(all_three),
    }

    # Overlap counts (interactions) — keep only stable, true, and stable ∩ true
    stable_and_true_inter = stable_interactions & inter_true

    inter_counts = {
        # "chosen_total": len(chosen_interactions),
        "stable_total": len(stable_interactions),
        "true_total": len(inter_true),
        # "chosen_and_stable": len(chosen_and_stable_inter),
        # "chosen_and_true": len(chosen_and_true_inter),
        "stable_and_true": len(stable_and_true_inter),
        # "all_three": len(all_three_inter),
    }

    # F1 metrics for mains and interactions (topN stable vs true)
    mains_metrics = compute_metrics(stable_mains, mains_true, is_interaction=False)
    interactions_metrics = compute_metrics(stable_interactions, inter_true, is_interaction=True)

    # Only plot Stable / True / Stable ∩ True
    labels = ["Stable", "True", "Stable ∩ True"]
    counts_m = [
        mains_counts["stable_total"],
        mains_counts["true_total"],
        mains_counts["stable_and_true"],
    ]
    counts_i = [
        inter_counts["stable_total"],
        inter_counts["true_total"],
        inter_counts["stable_and_true"],
    ]

    barplot_mains_png = os.path.join(out_dir, f"barplot_mains_{dataset}.png")
    barplot_inters_png = os.path.join(out_dir, f"barplot_interactions_{dataset}.png")
    plot_overlap_bars(labels, counts_m, f"{dataset} — Main Effects", barplot_mains_png)
    plot_overlap_bars(labels, counts_i, f"{dataset} — Interactions", barplot_inters_png)

    # Save summary JSON
    summary_out = {
        "dataset_stem": dataset,
        "tau": tau,
        "N_top_stable": N,
        "mains_counts": mains_counts,
        "interactions_counts": inter_counts,
        "mains_metrics": mains_metrics,
        "interactions_metrics": interactions_metrics,
        "stable_indices": {
            "mains": sorted(stable_mains),
            "interactions": [list(pair) for pair in stable_interactions]
        },
        "files": {
            "barplot_mains_png": barplot_mains_png,
            "barplot_interactions_png": barplot_inters_png,
            "mains_freq_csv": mains_freq_csv,
            "inter_freq_csv": inter_freq_csv
        }
    }
    summary_path_out = os.path.join(out_dir, f"stability_topN_summary_{dataset}.json")
    with open(summary_path_out, "w") as f:
        json.dump(summary_out, f, indent=2)
    print(f"Saved summary and plots for {dataset} in {out_dir}")

# --- Main loop over all tau thresholds ---
taus = [0.5, 0.55, 0.6, 0.65, 0.7]

for tau in taus:
    thresh_folder_name = f"stability_predcv_{tau}thresh"
    folder_path = os.path.join(base_results_dir, thresh_folder_name)
    out_dir_tau = os.path.join(base_results_dir, f"stability_topN_{N}_{tau}")
    os.makedirs(out_dir_tau, exist_ok=True)
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist, skipping.")
        continue
    summary_files = glob.glob(os.path.join(folder_path, "stability_summary_simulated_dataset_*.json"))
    for summary_file in summary_files:
        dataset = os.path.basename(summary_file).replace("stability_summary_", "").replace(".json", "")
        mains_freq_csv = os.path.join(folder_path, f"stability_mains_{dataset}.csv")
        inter_freq_csv = os.path.join(folder_path, f"stability_interactions_{dataset}.csv")
        if not (os.path.exists(mains_freq_csv) and os.path.exists(inter_freq_csv)):
            print(f"Missing frequency CSVs for {dataset}, skipping.")
            continue
        process_one(summary_file, mains_freq_csv, inter_freq_csv, out_dir_tau, N)
