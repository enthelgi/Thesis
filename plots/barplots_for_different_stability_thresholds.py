import os
import glob
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# List all tau folders you want to include
tau_folders = [
    #"stability_topN_50_0.5",
    "stability_topN_50_0.55",
    "stability_topN_50_0.6",
    "stability_topN_50_0.65",
    "stability_topN_50_0.7"
]
# tau_folders = [
#     "stability_predcv_0.5thresh",
#     "stability_predcv_0.55thresh",
#     "stability_predcv_0.6thresh",
#     "stability_predcv_0.65thresh",
#     "stability_predcv_0.7thresh"
# ]
base_dir = "C:/Users/enthe/Desktop/Thesis/results"

all_data = []
for tau_folder in tau_folders:
    results_dir = os.path.join(base_dir, tau_folder)
    summary_files = glob.glob(os.path.join(results_dir, "stability_topN_summary_simulated_dataset_*.json"))
    #summary_files = glob.glob(os.path.join(results_dir, "stability_summary_simulated_dataset_*.json"))

    print(f"Found {len(summary_files)} summary files in {tau_folder}")
    tau_label = tau_folder.split("_")[-1]
    for summary_file in summary_files:
        with open(summary_file) as f:
            data = json.load(f)
        dataset = data.get("dataset_stem", os.path.basename(summary_file))
        tau = data.get("tau", None)
        mains = data.get("mains_counts", {})
        interactions = data.get("interactions_counts", {})
        mains_stable = mains.get("stable_total", 0)
        mains_stable_and_true = mains.get("stable_and_true", 0)
        inter_stable = interactions.get("stable_total", 0)
        inter_stable_and_true = interactions.get("stable_and_true", 0)
        # Use tau_label for threshold
        threshold_label = tau_label
        all_data.append({
            "threshold": threshold_label,
            "dataset": dataset,
            "mains_stable_total": mains_stable,
            # "mains_stable_minus_true": mains_stable - mains_stable_and_true,
            # "inter_stable_total": inter_stable,
            # "inter_stable_minus_true": inter_stable - inter_stable_and_true
            "mains_stable_minus_true":  mains_stable_and_true/mains_stable if mains_stable > 0 else 0,
            "inter_stable_total": inter_stable,
            "inter_stable_minus_true": inter_stable_and_true/inter_stable if inter_stable > 0 else 0
        })

df = pd.DataFrame(all_data)
print("Columns in df:", df.columns)
if df.empty:
    print("No data loaded. DataFrame is empty. Exiting.")
    exit()

thresholds = sorted(df["threshold"].unique())
df_nonzero = df[(df["mains_stable_total"] > 0) & (df["inter_stable_total"] > 0)]

# --- Boxplots for all datasets ---
plt.figure(figsize=(14, 6))
plt.suptitle("All thresholds (tau)", fontsize=16)

plt.subplot(1, 2, 1)
sns.boxplot(
    data=df,
    x="threshold",
    y="mains_stable_minus_true",
    palette="Blues"
)
plt.xlabel("Stability Threshold (tau)")
plt.ylabel("Stable - (Stable & True) [Mains]")
plt.title("Mains: Stable minus (Stable & True)")

plt.subplot(1, 2, 2)
sns.boxplot(
    data=df,
    x="threshold",
    y="inter_stable_minus_true",
    palette="Oranges"
)
plt.xlabel("Stability Threshold (tau)")
plt.ylabel("Stable - (Stable & True) [Interactions]")
plt.title("Interactions: Stable minus (Stable & True)")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# --- Boxplots excluding datasets with 0 stable ---
plt.figure(figsize=(14, 6))
plt.suptitle("Excluding datasets with 0 stable", fontsize=16)

plt.subplot(1, 2, 1)
sns.boxplot(
    data=df_nonzero,
    x="threshold",
    y="mains_stable_minus_true",
    palette="Blues"
)
plt.xlabel("Stability Threshold (tau)")
plt.ylabel("Stable - (Stable & True) [Mains]")
plt.title("Mains: Stable minus (Stable & True)")

plt.subplot(1, 2, 2)
sns.boxplot(
    data=df_nonzero,
    x="threshold",
    y="inter_stable_minus_true",
    palette="Oranges"
)
plt.xlabel("Stability Threshold (tau)")
plt.ylabel("Stable - (Stable & True) [Interactions]")
plt.title("Interactions: Stable minus (Stable & True)")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

def bootstrap_ci(data, n_boot=1000, ci=95):
    data = np.array(data)
    boot_means = []
    for _ in range(n_boot):
        sample = np.random.choice(data, size=len(data), replace=True)
        boot_means.append(np.mean(sample))
    lower = np.percentile(boot_means, (100 - ci) / 2)
    upper = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return lower, upper

print("95% bootstrap confidence intervals for means:")
for t in thresholds:
    mains_vals = df[df["threshold"] == t]["mains_stable_minus_true"].values
    inters_vals = df[df["threshold"] == t]["inter_stable_minus_true"].values
    if len(mains_vals) > 0:
        mains_ci = bootstrap_ci(mains_vals)
        print(f"Threshold {t} - Mains: mean={np.mean(mains_vals):.2f}, 95% CI={mains_ci}")
    if len(inters_vals) > 0:
        inters_ci = bootstrap_ci(inters_vals)
        print(f"Threshold {t} - Inters: mean={np.mean(inters_vals):.2f}, 95% CI={inters_ci}")