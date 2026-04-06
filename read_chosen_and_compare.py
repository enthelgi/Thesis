import os
import glob
import json
import re
import pandas as pd

# Directory containing the chosen files
data_dir = "C:/Users/enthe/Desktop/Thesis/data/simulated_data_continuous"

# Patterns for each method
patterns = {
    "PL": os.path.join(data_dir, "chosen_continuous_simulated_dataset_*.json"),
    "iFORM": os.path.join(data_dir, "chosen_continuous_iform_simulated_dataset_*.json"),
    "BT": os.path.join(data_dir, "chosen_continuous_bt_simulated_dataset_*.json"),
}

# Regex to extract parameters from filename
param_re = re.compile(r"simulated_dataset_(\d+)_([0-9.]+)_(\d+)_([0-9.]+)")

rows = []

for method, pattern in patterns.items():
    for file in glob.glob(pattern):
        with open(file, "r") as f:
            data = json.load(f)
        # Extract parameters from filename
        m = param_re.search(file)
        if not m:
            continue
        def clean_num(s):
            return s.rstrip(".")
        n, noise_sd, p, rho = [clean_num(x) for x in m.groups()]
        metrics = data.get("metrics", {})
        row = {
            "Method": method,
            "n": int(n),
            "noise sd": float(noise_sd),
            "p": int(p),
            "rho": float(rho),
            "Main P": metrics.get("precision_mains", float('nan')),
            "Main R": metrics.get("recall_mains", float('nan')),
            "Main F1": metrics.get("f1_mains", float('nan')),
            "Inter P": metrics.get("precision_interactions", float('nan')),
            "Inter R": metrics.get("recall_interactions", float('nan')),
            "Inter F1": metrics.get("f1_interactions", float('nan')),
            #"MSE": metrics.get("mse_outer_mean", float('nan')),
            "MSE": metrics.get("mse_train_full", float('nan')),
            #"MSE": metrics.get("mse", float('nan')),

        }
        rows.append(row)

# Create DataFrame
df = pd.DataFrame(rows)

# Sort for nice output
df = df.sort_values(["n", "noise sd", "p", "rho", "Method"])

# Prepare LaTeX table
def fmt(x, col=None):
    if pd.isnull(x):
        return "0.000"
    if col in ["n", "p"]:
        return str(int(round(x)))
    if col == "noise sd":
        return f"{x:.1f}"
    if col == "rho":
        return f"{x:.1f}"
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)

# Columns for LaTeX (order as in your example, with new names)
cols = [
    "n", "noise sd", "p", "rho",
    "Main P", "Main R", "Main F1",
    "Inter P", "Inter R", "Inter F1"
]

# Ask user for method
valid_methods = ["PL", "iFORM", "BT"]
method = ""
while method not in valid_methods:
    method = input(f"Which method do you want the table for? ({'/'.join(valid_methods)}): ").strip()
    if method not in valid_methods:
        print("Invalid method. Please enter one of:", ", ".join(valid_methods))

df_method = df[df["Method"] == method].copy()
if df_method.empty:
    print(f"No data found for method {method}.")
    exit(1)

df_latex = df_method[cols].copy()

# Format for LaTeX
latex_rows = []
for _, row in df_latex.iterrows():
    latex_rows.append(" & ".join(fmt(row[col], col) for col in cols) + r" \\")

latex_table = (
    r"\setlength{\tabcolsep}{4pt} %% default is 6pt" "\n"
    r"\renewcommand{\arraystretch}{0.70}" "\n"
    r"\begin{table}[htbp]" "\n"
    r"\centering" "\n"
    rf"\caption{{{method}: Evaluation Metrics for Main and Interaction Effects. Main refers to main effects and Inter refers to interaction effects.}}" "\n"
    rf"\label{{tab:{method.lower()}_evaluation_metrics}}" "\n"
    r"\resizebox{\textwidth}{!}{%%" "\n"
    r"\begin{tabular}{rrrrrrrrrr}" "\n"
    r"\toprule" "\n"
    r"$n$ & noise sd & $p$ & $\rho$ & Main P & Main R & Main F1 & Inter P & Inter R & Inter F1 \\" "\n"
    r"\midrule" "\n"
    + "\n".join(latex_rows) + "\n"
    r"\bottomrule" "\n"
    r"\end{tabular}%%" "\n"
    r"}" "\n"
    r"\end{table}"
)

print(latex_table)

# --- Summary Table (averages per method, only score columns + MSE) ---
summary_cols = [
    "rho",
    "Main P", "Main R", "Main F1",
    "Inter P", "Inter R", "Inter F1",
    "MSE"

]

summary_rows = []
for m in valid_methods:
    if m == method:
        df_m = df[df["Method"] == m]
        if not df_m.empty:
            vals = [fmt(df_m[c].mean(), c) for c in summary_cols]
        else:
            vals = [" - "] * len(summary_cols)
    else:
        vals = [" - "] * len(summary_cols)
    summary_rows.append(" & ".join(vals) + r" \\")

summary_latex_table = (
    r"\setlength{\tabcolsep}{4pt} %% default is 6pt" "\n"
    r"\renewcommand{\arraystretch}{0.70}" "\n"
    r"\begin{table}[htbp]" "\n"
    r"\centering" "\n"
    r"\caption{Average Evaluation Metrics for Main and Interaction Effects across all datasets.}" "\n"
    r"\label{tab:average_evaluation_metrics}" "\n"
    r"\resizebox{\textwidth}{!}{%%" "\n"
    r"\begin{tabular}{lrrrrrrrr}" "\n"
    r"\toprule" "\n"
    r"Method & $\rho$ & Main P & Main R & Main F1 & Inter P & Inter R & Inter F1 & MSE \\" "\n"
    r"\midrule" "\n"
    f"PL & {summary_rows[0]}\n"
    f"iFORM & {summary_rows[1]}\n"
    f"BT & {summary_rows[2]}"
    r"\bottomrule" "\n"
    r"\end{tabular}%%" "\n"
    r"}" "\n"
    r"\end{table}"
)

print(summary_latex_table)