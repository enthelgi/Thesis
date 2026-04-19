import json
import pandas as pd

# Filepaths
json_path = "C:/Users/enthe/Desktop/Thesis/results/tgs_results/stability_selected_tgs_thresh_0.65_balanced.json"
mains_csv = "C:/Users/enthe/Desktop/Thesis/results/tgs_results/stability_mains_tgs_balanced.csv"
inter_csv = "C:/Users/enthe/Desktop/Thesis/results/tgs_results/stability_interactions_tgs_balanced.csv"

# Load selected features/interactions
with open(json_path) as f:
    selected = json.load(f)
selected_mains = selected["selected_mains"]
selected_inters = selected["selected_interactions"]

# Load frequencies
mains_df = pd.read_csv(mains_csv, index_col=0)
inter_df = pd.read_csv(inter_csv)

# Prepare main effects table, sorted by frequency
main_rows = []
for feat in selected_mains:
    freq = mains_df.loc[feat, "freq"]
    main_rows.append((feat, freq))
main_rows.sort(key=lambda x: x[1], reverse=True)

# Prepare interactions table, sorted by frequency
name_to_idx = {name: i for i, name in enumerate(mains_df.index)}
inter_rows = []
for a, b in selected_inters:
    i, j = name_to_idx[a], name_to_idx[b]
    freq = inter_df[((inter_df["i"] == i) & (inter_df["j"] == j)) | ((inter_df["i"] == j) & (inter_df["j"] == i))]["freq"].values
    freq_val = freq[0] if len(freq) > 0 else 0.0
    inter_rows.append((f"{a} -- {b}", freq_val))
inter_rows.sort(key=lambda x: x[1], reverse=True)

# Print LaTeX table
print(r"""\begin{table}[htbp]
\centering
\caption{Selected Main Effects and Interactions with Selection Frequency ($\tau=0.65$, balanced, sorted by frequency)}
\label{tab:selected_effects_freq_sorted}
\begin{tabular}{ll}
\toprule
Main Effect & Frequency \\
\midrule""")
for feat, freq in main_rows:
    print(f"{feat} & {freq:.2f} \\\\")
print(r"""\bottomrule
\end{tabular}

\vspace{1em}

\begin{tabular}{ll}
\toprule
Interaction & Frequency \\
\midrule""")
for inter, freq in inter_rows:
    print(f"{inter} & {freq:.2f} \\\\")
print(r"""\bottomrule
\end{tabular}
\end{table}""")