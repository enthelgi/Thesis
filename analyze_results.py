#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze stability summaries across all datasets and plot n×p grids per rho.

- Reads every stability_summary_*.json in CONFIG["DATA_DIR"].
- Builds a tidy DataFrame with parameters (n, prev, p, rho) and metrics for mains/interactions.
- Prints aggregate statistics (overall and grouped by rho).
- Creates FacetGrid scatter plots (n vs p) colored by the “All three” count (and also saves a normalized version).
- Makes horizontal boxplots of interaction F1 grouped by rho (one box per rho).
"""

import os
import re
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ========== CONFIG ==========
CONFIG = {
    "DATA_DIR": r"C:/Users/enthe/Desktop/Thesis/results/stability_topN_50_0.65",
    "OUT_DIR": r"C:/Users/enthe/Desktop/Thesis/results/stability_statistics_topN_50_0.65",
    "FIG_DPI": 220,
    "RHO_ORDER": [0.0, 0.5, 0.8],
    "USE_RATE_TOO": False,           # also plot normalized color = all_three / true_total
    "CONTEXT": "talk",
    "STYLE": "whitegrid",
}

sns.set_theme(style=CONFIG["STYLE"], context=CONFIG["CONTEXT"])

# ---------- Helpers ----------
#STEM_RE = re.compile(r"^stability_summary_(simulated_dataset_(\d+)_([0-9.]+)_(\d+)_([0-9.]+))\.json$")
#STEM_RE = re.compile(r"^stability_summary_(simulated_dataset_\d+_[\d.]+_\d+_[\d.]+)\.json$")
STEM_RE = re.compile(r"^stability_topN_summary_(simulated_dataset_\d+_[\d.]+_\d+_[\d.]+)\.json$")

def parse_stem_from_summary(filename: str) -> Tuple[str, int, float, int, float]:
    m = STEM_RE.match(filename)
    if not m:
        raise ValueError(f"Unrecognized summary filename: {filename}")
    stem = m.group(1)
    # Extract numbers from the stem
    parts = stem.split("_")
    n = int(parts[2])
    prev = float(parts[3])
    p = int(parts[4])
    rho = float(parts[5])
    return stem, n, prev, p, rho


def load_all_summaries(data_dir: str) -> List[Dict]:
    rows = []
    for fname in os.listdir(data_dir):
        if not fname.startswith("stability_topN_summary_") or not fname.endswith(".json"):
            continue
        try:
            stem, n, prev, p, rho = parse_stem_from_summary(fname)
        except Exception as e:
            print(f"Skipping {fname}: {e}")
            continue

        fpath = os.path.join(data_dir, fname)
        with open(fpath, "r") as f:
            s = json.load(f)

        thresholds = s.get("thresholds", {})
        implied_pfer_main = thresholds.get("implied_pfer_main", np.nan)
        implied_pfer_inter = thresholds.get("implied_pfer_inter", np.nan)
        stability_method = s.get("stability_method", "stability")
        pair_agg = s.get("pair_agg", None)

        # For each type, extract counts and metrics from the new format
        for typ in ("mains", "interactions"):
            counts = s.get(f"{typ}_counts", {})
            metrics = s.get(f"{typ}_metrics", {})  # If you have metrics blocks, else leave as {}

            chosen_total = counts.get("chosen_total", np.nan)
            stable_total = counts.get("stable_total", np.nan)
            true_total = counts.get("true_total", np.nan)
            chosen_and_stable = counts.get("chosen_and_stable", np.nan)
            chosen_and_true = counts.get("chosen_and_true", np.nan)
            stable_and_true = counts.get("stable_and_true", np.nan)
            
            # NEW: accept either "all_three" (new) or "chosen_and_stable_and_true" (old)
            all_three = counts.get("all_three",
                                counts.get("chosen_and_stable_and_true", np.nan))
            row = dict(
                stem=stem, n=n, prev=prev, p=p, rho=rho, typ=typ,
                stability_method=stability_method,
                pair_agg=pair_agg,
                pi_main=thresholds.get("pi_main", np.nan),
                pi_inter=thresholds.get("pi_inter", np.nan),
                implied_pfer_main=implied_pfer_main,
                implied_pfer_inter=implied_pfer_inter,
                chosen_total=chosen_total,
                stable_total=stable_total,
                true_total=true_total,
                chosen_and_stable=chosen_and_stable,
                chosen_and_true=chosen_and_true,
                stable_and_true=stable_and_true,
                all_three=all_three,
                F1=metrics.get("F1", np.nan),
                precision=metrics.get("precision", np.nan),
                recall=metrics.get("recall", np.nan),
                TP=metrics.get("TP", np.nan),
                FP=metrics.get("FP", np.nan),
                FN=metrics.get("FN", np.nan),
                n_stable=metrics.get("n_stable", np.nan),
            )
            row["all_three_rate"] = (all_three / true_total) if (isinstance(true_total, (int, float)) and true_total) else np.nan
            row["stable_true_rate"] = (stable_and_true / true_total) if (isinstance(true_total, (int, float)) and true_total) else np.nan
            rows.append(row)
    return rows


def print_aggregates(df: pd.DataFrame):
    print("\n=== Overall means by type ===")
    print(df.groupby("typ")[["F1", "precision", "recall", "all_three", "all_three_rate"]].median().round(3))

    print("\n=== Means by type and rho ===")
    print(df.groupby(["typ", "rho"])[["F1", "precision", "recall", "all_three", "all_three_rate"]].median().round(3))

    print("\n=== Count of datasets by type and rho ===")
    print(df.groupby(["typ", "rho"]).size())


# def facet_scatter(df: pd.DataFrame, color_col: str, title_suffix: str, out_png: str):
#     g = sns.FacetGrid(
#         df, col="rho", hue=color_col, col_order=CONFIG["RHO_ORDER"],
#         sharex=False, sharey=False, height=4.0, aspect=1.2,
#         palette="viridis"
#     )
#     g.map_dataframe(sns.scatterplot, x="n", y="p", s=120, edgecolor="black", linewidth=0.5)

#     import matplotlib as mpl
#     v = df[color_col].values.astype(float)
#     v = v[np.isfinite(v)]
#     vmin = float(np.nanmin(v)) if v.size else 0.0
#     vmax = float(np.nanmax(v)) if v.size else 100.0 if "percent" in color_col else 1.0
#     norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
#     sm = mpl.cm.ScalarMappable(cmap="viridis", norm=norm)
#     sm.set_array([])
#     g.figure.subplots_adjust(top=0.82, right=0.88)
#     cax = g.figure.add_axes([0.90, 0.25, 0.02, 0.5])
#     label = "Recovered %" if "percent" in color_col else color_col.replace("_", " ")
#     g.figure.colorbar(sm, cax=cax, label=label)

#     g.set_axis_labels("n", "p")
#     g.set_titles("ρ = {col_name}")
#     g.figure.suptitle(f"Dataset grid colored by {title_suffix}", y=0.98)
#     for ax in np.ravel(g.axes):
#         if ax is None:
#             continue
#         ax.grid(True, axis="both", which="major", alpha=0.2)
#         ax.set_xticks(sorted(df["n"].dropna().unique()))
#         ax.set_yticks(sorted(df["p"].dropna().unique()))
#     plt.savefig(out_png, dpi=CONFIG["FIG_DPI"], bbox_inches="tight")
#     plt.close()
#     print(f"Saved {out_png}")

def facet_scatter(df: pd.DataFrame, color_col: str, title_suffix: str, out_png: str):
    import matplotlib as mpl

    # Compute color scale globally across facets
    v = pd.to_numeric(df[color_col], errors="coerce").values.astype(float)
    v = v[np.isfinite(v)]
    vmin = float(np.nanmin(v)) if v.size else 0.0
    vmax = float(np.nanmax(v)) if v.size else (100.0 if "percent" in color_col else 1.0)

    # Build grid WITHOUT hue
    g = sns.FacetGrid(
        df, col="rho", col_order=CONFIG["RHO_ORDER"],
        sharex=False, sharey=False, height=4.0, aspect=1.2
    )

    # Map a scatter that uses a continuous hue
    g.map_dataframe(
        sns.scatterplot,
        x="n", y="p",
        hue=color_col,
        s=120, edgecolor="black", linewidth=0.5,
        palette=sns.color_palette("viridis", as_cmap=True),
        hue_norm=(vmin, vmax),
        legend=False,  # we’ll add a colorbar instead
    )

    # Shared colorbar
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = mpl.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    g.figure.subplots_adjust(top=0.82, right=0.88)
    cax = g.figure.add_axes([0.90, 0.25, 0.02, 0.5])
    label = "Recovered %" if "percent" in color_col else color_col.replace("_", " ")
    g.figure.colorbar(sm, cax=cax, label=label)

    # Cosmetics
    g.set_axis_labels("n", "p")
    g.set_titles("ρ = {col_name}")
    g.figure.suptitle(f"Dataset grid colored by {title_suffix}", y=0.98)

    for ax in np.ravel(g.axes):
        if ax is None:
            continue
        ax.grid(True, axis="both", which="major", alpha=0.2)
        # keep ticks tidy even if some facets are empty
        if "n" in df:
            xt = sorted(pd.unique(df["n"].dropna()))
            if len(xt) <= 15:
                ax.set_xticks(xt)
        if "p" in df:
            yt = sorted(pd.unique(df["p"].dropna()))
            if len(yt) <= 15:
                ax.set_yticks(yt)

    plt.savefig(out_png, dpi=CONFIG["FIG_DPI"], bbox_inches="tight")
    plt.close()
    print(f"Saved {out_png}")



def boxplot_interaction_f1(df: pd.DataFrame, out_png: str):
    """
    Horizontal boxplots of interaction F1 by rho (one box per rho),
    using all datasets (typ == 'interactions'), with jittered points overlay.
    """
    dfi = df[(df["typ"] == "interactions") & (~df["F1"].isna())].copy()
    if dfi.empty:
        print("No interaction F1 values to plot.")
        return

    # Keep rho order consistent
    dfi["rho"] = pd.Categorical(dfi["rho"], categories=CONFIG["RHO_ORDER"], ordered=True)

    plt.figure(figsize=(8, 4.5))
    ax = sns.boxplot(
        data=dfi,
        y="rho", x="F1",
        order=CONFIG["RHO_ORDER"],
        orient="h",
        whis=(5, 95),
        showcaps=True,
        showfliers=False
    )
    sns.stripplot(
        data=dfi,
        y="rho", x="F1",
        order=CONFIG["RHO_ORDER"],
        orient="h",
        size=4, alpha=0.35, linewidth=0, color="black", jitter=0.15
    )

    ax.set_title("Interaction recovery F1 by ρ")
    ax.set_xlabel("F1 (stable vs true)")
    ax.set_ylabel("ρ")
    ax.set_xlim(0, 1)
    plt.tight_layout()
    plt.savefig(out_png, dpi=CONFIG["FIG_DPI"])
    plt.close()
    print(f"Saved {out_png}")

def scatter_rho_prev(df: pd.DataFrame, color_col: str, title_suffix: str, out_png: str):
    dfx = df.copy()
    dfx = dfx[(~dfx["rho"].isna()) & (~dfx["prev"].isna())].copy()
    if dfx.empty:
        print(f"No rows to plot for {title_suffix}.")
        return

    dfx["rho"] = dfx["rho"].astype(float)
    dfx["rho"] = pd.Categorical(dfx["rho"], categories=CONFIG["RHO_ORDER"], ordered=True)

    agg_cols = [c for c in dfx.columns if dfx[c].dtype.kind in "fcib" and c not in ("rho", "prev")]
    grouped = (
        dfx.groupby(["typ", "rho", "prev"], observed=True)[agg_cols]
           .mean(numeric_only=True)
           .reset_index()
    )
    counts = (
        dfx.groupby(["typ", "rho", "prev"], observed=True)
           .size()
           .rename("n_datasets")
           .reset_index()
    )
    da = grouped.merge(counts, on=["typ", "rho", "prev"], how="left")

    import matplotlib as mpl
    v = da[color_col].values.astype(float)
    v = v[np.isfinite(v)]
    vmin = float(np.nanmin(v)) if v.size else 0.0
    vmax = float(np.nanmax(v)) if v.size else 100.0 if "percent" in color_col else 1.0
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("viridis")
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    size = 60 * np.sqrt(da["n_datasets"].fillna(1).astype(float))

    plt.figure(figsize=(7.5, 5.0))
    ax = plt.gca()
    sc = ax.scatter(
        x=da["rho"].cat.codes,
        y=da["prev"],
        c=da[color_col],
        s=size,
        edgecolor="black",
        linewidth=0.5,
        cmap=cmap,
        norm=norm,
        alpha=0.9,
    )

    ax.set_title(f"ρ × prevalence grid colored by {title_suffix}")
    ax.set_xlabel("ρ")
    ax.set_ylabel("Prevalence")
    ax.set_xticks(range(len(CONFIG["RHO_ORDER"])))
    ax.set_xticklabels([str(r) for r in CONFIG["RHO_ORDER"]])
    prev_vals = sorted(da["prev"].dropna().unique().tolist())
    ax.set_yticks(prev_vals)
    ax.grid(True, axis="both", which="major", alpha=0.25)

    label = "Recovered %" if "percent" in color_col else color_col.replace("_", " ")
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(label)

    for_sizes = sorted(da["n_datasets"].dropna().unique())
    reps = [1, 2, max(for_sizes) if len(for_sizes) else 1]
    reps = sorted(set(int(x) for x in reps if x >= 1))[:3]
    handles = [
        plt.scatter([], [], s=60 * np.sqrt(k), edgecolor="black", linewidth=0.5, facecolor="none")
        for k in reps
    ]
    labels = [f"{k} dataset" + ("" if k == 1 else "s") for k in reps]
    if handles:
        ax.legend(handles, labels, title="Avg. over", loc="best", frameon=True)

    plt.tight_layout()
    plt.savefig(out_png, dpi=CONFIG["FIG_DPI"], bbox_inches="tight")
    plt.close()
    print(f"Saved {out_png}")



def main():
    #print("Looking in:", CONFIG["DATA_DIR"])
    #print("Files found:", os.listdir(CONFIG["DATA_DIR"]))
    rows = load_all_summaries(CONFIG["DATA_DIR"])
    if not rows:
        print("No stability_summary_*.json files found.")
        return
    df = pd.DataFrame(rows)

    # Clean rho types/order for pretty facets
    df["rho"] = df["rho"].astype(float)
    df["rho"] = pd.Categorical(df["rho"], categories=CONFIG["RHO_ORDER"], ordered=True)
    df["all_three_percent"] = (
        df["all_three"] / df["true_total"] * 100
        if "all_three" in df and "true_total" in df else np.nan
    )
    df["stable_true_percent"] = (
        df["stable_and_true"] / df["true_total"] * 100
        if "stable_and_true" in df and "true_total" in df else np.nan
    )
    # Print aggregates
    print_aggregates(df)

    # Save a CSV for further analysis
    out_csv = os.path.join(CONFIG["OUT_DIR"], "aggregate_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved aggregated results → {out_csv}")

    # Boxplots of interaction F1 by rho (across all datasets)
    out_box = os.path.join(CONFIG["OUT_DIR"], "box_interactions_F1_by_rho.png")
    boxplot_interaction_f1(df, out_box)

    # Plots for mains and interactions separately
    for typ in ("mains", "interactions"):
        dft = df[df["typ"] == typ].copy()
        if dft.empty:
            continue
        # Raw all_three count
        out_png_count = os.path.join(CONFIG["OUT_DIR"], f"grid_{typ}_allthree.png")
        facet_scatter(dft, color_col="all_three", title_suffix=f'“All three” count ({typ})', out_png=out_png_count)
        out_png_allthree_pct = os.path.join(CONFIG["OUT_DIR"], f"grid_{typ}_allthree_percent.png")
        facet_scatter(
            dft,
            color_col="all_three_percent",
            title_suffix=f'Recovered % (“All three”, {typ})',
            out_png=out_png_allthree_pct
        )

        out_rp_allthree_pct = os.path.join(CONFIG["OUT_DIR"], f"grid_rho_prev_{typ}_allthree_percent.png")
        scatter_rho_prev(
            dft,
            color_col="all_three_percent",
            title_suffix=f'Recovered % (“All three”, {typ})',
            out_png=out_rp_allthree_pct
        )
        out_png_stable_true_pct = os.path.join(CONFIG["OUT_DIR"], f"grid_{typ}_stable_true_percent.png")
        facet_scatter(
            dft,
            color_col="stable_true_percent",
            title_suffix=f'Recovered % (Stable & True, {typ})',
            out_png=out_png_stable_true_pct
        )

        # ρ × prevalence grid for stable_and_true percent
        out_rp_stable_true_pct = os.path.join(CONFIG["OUT_DIR"], f"grid_rho_prev_{typ}_stable_true_percent.png")
        scatter_rho_prev(
            dft,
            color_col="stable_true_percent",
            title_suffix=f'Recovered % (Stable & True, {typ})',
            out_png=out_rp_stable_true_pct
        )

        # Normalized rate
        if CONFIG["USE_RATE_TOO"]:
            out_png_rate = os.path.join(CONFIG["OUT_DIR"], f"grid_{typ}_allthree_rate.png")
            facet_scatter(dft, color_col="all_three_rate", title_suffix=f'“All three” rate ({typ})', out_png=out_png_rate)
    # -------- Add these calls inside main(), after the existing grid plots --------
    # (Search for the loop: `for typ in ("mains", "interactions"):` and append the following inside it.)

        # === New ρ × prevalence grids ===
        # We call on a per-typ slice to keep behavior consistent.
        # (The function aggregates multiple datasets at identical (rho, prev) by averaging.)

        # Count-based color
        out_rp_count = os.path.join(CONFIG["OUT_DIR"], f"grid_rho_prev_{typ}_allthree.png")
        scatter_rho_prev(dft, color_col="all_three", title_suffix=f'“All three” count ({typ})', out_png=out_rp_count)

        # Normalized rate color
        if CONFIG["USE_RATE_TOO"]:
            out_rp_rate = os.path.join(CONFIG["OUT_DIR"], f"grid_rho_prev_{typ}_allthree_rate.png")
            scatter_rho_prev(dft, color_col="all_three_rate", title_suffix=f'“All three” rate ({typ})', out_png=out_rp_rate)



if __name__ == "__main__":
    main()
