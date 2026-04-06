import os
import glob
import json

results_dir = "C:/Users/enthe/Desktop/Thesis/results"
summary_files = glob.glob(os.path.join(results_dir, "stability_topN_20_0.65", "stability_topN_summary_simulated_dataset_*.json"))

rows = []
for f in summary_files:
    with open(f) as jf:
        data = json.load(jf)
    # Parse dataset stem for n, prev, p, rho
    stem = data.get("dataset_stem", "")
    try:
        parts = stem.split("_")
        n = int(parts[2])
        prev = float(parts[3])
        p = int(parts[4])
        rho = float(parts[5])
    except Exception:
        n, prev, p, rho = "-", "-", "-", "-"
    # Main metrics
    mm = data.get("mains_metrics", {})
    main_p = mm.get("precision", "-")
    main_r = mm.get("recall", "-")
    main_f1 = mm.get("F1", "-")
    # Inter metrics
    im = data.get("interactions_metrics", {})
    inter_p = im.get("precision", "-")
    inter_r = im.get("recall", "-")
    inter_f1 = im.get("F1", "-")
    # Format: if metric is 0, show 0.000; if missing or 0.0, show '-'
    def fmt(x):
        if x == "-" or x is None:
            return "-"
        try:
            x = float(x)
            return "-" if x == 0 else f"{x:.3f}"
        except Exception:
            return "-"
    rows.append([
        n, f"{prev:.2f}", p, f"{rho:.1f}",
        fmt(main_p), fmt(main_r), fmt(main_f1),
        fmt(inter_p), fmt(inter_r), fmt(inter_f1)
    ])

print(r"""\setlength{\tabcolsep}{4pt} % default is 6pt
\renewcommand{\arraystretch}{0.65}
\begin{table}[htbp]
\centering
\caption{BT: Evaluation Metrics for Main and Interaction Effects. Main refers to main effects and Inter refers to interaction effects.}
\label{tab:bt_evaluation_metrics}
\resizebox{\textwidth}{!}{%
\begin{tabular}{rrrrrrrrrr}
\toprule
$n$ & $prev$ & $p$ & $\rho$ & Main P & Main R & Main F1 & Inter P & Inter R & Inter F1 \\
\midrule""")

for row in sorted(rows):
    print(" & ".join(str(x) for x in row) + r" \\")
print(r"""\bottomrule
\end{tabular}%
}
\end{table}""")


# import os
# import glob
# import json

# results_dir = "C:/Users/enthe/Desktop/Thesis/results"
# summary_files = glob.glob(os.path.join(results_dir, "fit_predcv", "best_model_simulated_dataset_*.json"))

# rows = []
# for f in summary_files:
#     with open(f) as jf:
#         data = json.load(jf)
#     # Parse dataset stem for n, prev, p, rho
#     stem = os.path.basename(f).replace("best_model_simulated_dataset_", "").replace(".json", "")
#     try:
#         parts = stem.split("_")
#         n = int(parts[0])
#         prev = float(parts[1])
#         p = int(parts[2])
#         rho = float(parts[3])
#     except Exception:
#         n, prev, p, rho = "-", "-", "-", "-"
#     # Main metrics
#     main_p = data.get("precision_main", "-")
#     main_r = data.get("recall_main", "-")
#     main_f1 = data.get("F1_main", "-")
#     # Inter metrics
#     inter_p = data.get("precision_inter", "-")
#     inter_r = data.get("recall_inter", "-")
#     inter_f1 = data.get("F1_inter", "-")
#     # Format: if metric is 0, show 0.000; if missing or 0.0, show '-'
#     def fmt(x):
#         if x == "-" or x is None:
#             return "-"
#         try:
#             x = float(x)
#             return "-" if x == 0 else f"{x:.3f}"
#         except Exception:
#             return "-"
#     rows.append([
#         n, f"{prev:.2f}", p, f"{rho:.1f}",
#         fmt(main_p), fmt(main_r), fmt(main_f1),
#         fmt(inter_p), fmt(inter_r), fmt(inter_f1)
#     ])

# print(r"""\setlength{\tabcolsep}{4pt} % default is 6pt
# \renewcommand{\arraystretch}{0.65}
# \begin{table}[htbp]
# \centering
# \caption{BT: Evaluation Metrics for Main and Interaction Effects (fit_predcv results). Main refers to main effects and Inter refers to interaction effects.}
# \label{tab:bt_evaluation_metrics_predcv}
# \resizebox{\textwidth}{!}{%
# \begin{tabular}{rrrrrrrrrr}
# \toprule
# $n$ & $prev$ & $p$ & $\rho$ & Main P & Main R & Main F1 & Inter P & Inter R & Inter F1 \\
# \midrule""")

# for row in sorted(rows):
#     print(" & ".join(str(x) for x in row) + r" \\")
# print(r"""\bottomrule
# \end{tabular}%
# }
# \end{table}""")