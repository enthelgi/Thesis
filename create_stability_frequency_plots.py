import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

base = r"c:\Users\enthe\Desktop\Thesis\results\tgs_results"

files = {
    "TGS_balanced": {
        "mains": os.path.join(base, "stability_mains_tgs_balanced.csv"),
        "inter": os.path.join(base, "stability_interactions_tgs_balanced.csv"),
    },
    "TGS_imbalanced": {
        "mains": os.path.join(base, "stability_mains_tgs_imbalanced.csv"),
        "inter": os.path.join(base, "stability_interactions_tgs_imbalanced.csv"),
    },
    "TGS_L1_balanced": {
        "mains": os.path.join(base, "stability_mains_tgs_l1logistic_balanced.csv"),
        "inter": os.path.join(base, "stability_interactions_tgs_l1logistic_balanced.csv"),
    },
    "TGS_L1_imbalanced": {
        "mains": os.path.join(base, "stability_mains_tgs_l1logistic_imbalanced.csv"),
        "inter": os.path.join(base, "stability_interactions_tgs_l1logistic_imbalanced.csv"),
    },
    "TGS_iFORM_balanced": {
        "mains": os.path.join(base, "stability_mains_tgs_iform_balanced.csv"),
        "inter": os.path.join(base, "stability_interactions_tgs_iform_balanced.csv"),
    },
    "TGS_iFORM_imbalanced": {
        "mains": os.path.join(base, "stability_mains_tgs_iform_imbalanced.csv"),
        "inter": os.path.join(base, "stability_interactions_tgs_iform_imbalanced.csv"),
    },
}

def get_model_name(key):
    if "iFORM" in key:
        return "iFORM"
    elif "L1" in key:
        return "L1"
    else:
        return "PL"

def plot_inter_heatmap(df, title, outname):
    max_idx = max(df['i'].max(), df['j'].max())
    mat = np.zeros((max_idx+1, max_idx+1))
    for _, row in df.iterrows():
        mat[int(row['i']), int(row['j'])] = row['freq']
        mat[int(row['j']), int(row['i'])] = row['freq']
    mask = np.triu(np.ones_like(mat, dtype=bool), k=1)
    plt.figure(figsize=(8, 6))
    sns.heatmap(mat, cmap="viridis", vmin=0, vmax=1, square=True, cbar_kws={'label': 'Frequency'},
                mask=mask, xticklabels=True, yticklabels=True)
    plt.title(title)
    plt.xlabel("Feature index (j)")
    plt.ylabel("Feature index (i)")
    plt.tight_layout()
    plt.savefig(outname)
    plt.close()

def plot_mains_bar(df, title, outname):
    if df.index.dtype == 'O':
        x = np.arange(len(df))
    else:
        x = df.index.values
    plt.figure(figsize=(10, 4))
    plt.bar(x, df['freq'])
    plt.ylim(0, 1)
    plt.title(title)
    plt.xlabel("Feature index")
    plt.ylabel("Frequency")
    if len(x) > 30:
        step = max(1, len(x)//20)
        plt.xticks(np.arange(0, len(x), step))
    else:
        plt.xticks(x)
    plt.tight_layout()
    plt.savefig(outname)
    plt.close()

for key, paths in files.items():
    model_name = get_model_name(key)
    # Mains
    mains = pd.read_csv(paths["mains"], index_col=0)
    plot_mains_bar(
        mains,
        f"Main Frequencies: {model_name} {key.replace('_', ' ')}",
        f"mains_{key}.png"
    )
    # Interactions
    inter = pd.read_csv(paths["inter"])
    plot_inter_heatmap(
        inter,
        f"Interaction Frequencies: {model_name} {key.replace('_', ' ')}",
        f"inter_heatmap_{key}.png"
    )

print("Plots saved as mains_*.png and inter_heatmap_*.png")