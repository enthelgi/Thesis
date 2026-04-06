import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load mains ---
mains_path = "C:/Users/enthe/Desktop/Thesis/results/tgs_results/stability_mains_tgs_balanced.csv"
mains_df = pd.read_csv(mains_path, index_col=0)
main_names = mains_df.index.tolist()
main_freqs = mains_df["freq"].values

# --- Load interactions ---
inter_path = "C:/Users/enthe/Desktop/Thesis/results/tgs_results/stability_interactions_tgs_balanced.csv"
inter_df = pd.read_csv(inter_path)
#print(inter_df)

# If your interaction indices are 1-based in the CSV, uncomment the next two lines:
# inter_df["i"] = inter_df["i"].astype(int) - 1
# inter_df["j"] = inter_df["j"].astype(int) - 1

p = len(main_names)

# Build full symmetric frequency matrix
heatmap_matrix = np.zeros((p, p), dtype=float)
for _, row in inter_df.iterrows():
    i, j, freq = int(row["i"]), int(row["j"]), float(row["freq"])
    if 0 <= i < p and 0 <= j < p:
        heatmap_matrix[i, j] = freq
        heatmap_matrix[j, i] = freq  # symmetric

# ---------------------------
# Plot 1: bar plot for mains
# ---------------------------
plt.figure(figsize=(max(8, p // 2), 6))
plt.bar(main_names, main_freqs, color="skyblue", edgecolor="black")
plt.axhline(y=0.65, color="red", linestyle="--", label="Threshold = 0.65")
plt.xticks(rotation=90)
plt.xlabel("Feature")
plt.ylabel("Selection Frequency")
plt.title("Selection Frequency of Main Effects")
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------------
# Plot 2: interactions heatmap — LOWER triangle only, scaled to existing max
# -------------------------------------------------------------------------
# Keep only lower triangle (exclude diagonal) in a NaN matrix so NaNs don't affect vmax
lower_matrix = np.full((p, p), np.nan, dtype=float)
# fill i > j entries from heatmap_matrix
lower_indices = np.tril_indices(p, k=-1)
lower_matrix[lower_indices] = heatmap_matrix[lower_indices]

# Compute vmax from existing (non-NaN) values shown
if np.isnan(lower_matrix).all():
    print("No interaction frequencies found for lower triangle.")
    vmax_full = 1.0  # fallback to make a valid colorbar
else:
    vmax_full = float(np.nanmax(lower_matrix))

# Mask the upper triangle (including diagonal)
mask_full = np.triu(np.ones_like(lower_matrix, dtype=bool), k=0)

plt.figure(figsize=(max(10, p // 2), max(8, p // 2)))
sns.heatmap(
    lower_matrix,
    xticklabels=main_names,
    yticklabels=main_names,
    cmap="viridis",
    vmin=0,
    vmax=vmax_full,
    mask=mask_full,
    cbar=True,
    square=True
)
plt.title("Selection Frequency Heatmap of Interactions")
plt.xlabel("Feature")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# --------------------------------------------------------------------------------------
# Plot 3: selected interactions — LOWER triangle only, thresholded, scaled to existing max
# --------------------------------------------------------------------------------------
threshold = 0.65
selected_matrix = np.full((p, p), np.nan, dtype=float)


for _, row in inter_df.iterrows():
    i, j, freq = int(row["i"]), int(row["j"]), float(row["freq"])
    print(freq)
    if 0 <= i < p and 0 <= j < p and freq >= threshold:
        selected_matrix[i, j] = freq
#print(selected_matrix)
# Compute vmax from the actually shown values
if np.isnan(selected_matrix).all():
    print(f"No selected interactions above threshold ({threshold}).")
    vmax_sel = 1.0  # fallback; plot will be empty but valid
else:
    vmax_sel = float(np.nanmax(selected_matrix))

# Mask the upper triangle (including diagonal)
#mask_sel = np.triu(np.ones_like(selected_matrix, dtype=bool), k=0)

plt.figure(figsize=(max(10, p // 2), max(8, p // 2)))
sns.heatmap(
    selected_matrix,
    xticklabels=main_names,
    yticklabels=main_names,
    cmap="viridis",
    vmin=0,
    vmax=vmax_sel,
    #mask=mask_sel,
    cbar=True,
    square=True
)
plt.title(f"Selected Interactions (freq ≥ {threshold})")
plt.xlabel("Feature")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
