import pandas as pd
import numpy as np
import json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, auc, average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt
import os

# --- Load data ---
csv_path = "data/simulated_data/simulated_dataset_250_0.5_20_0.0.csv"
summary_path = "results/stability/stability_summary_simulated_dataset_250_0.5_20_0.0.json" 
#top50 gives better predictive performance than 20
summary_path = "C:/Users/enthe/Desktop/Thesis/results/stability_topN_50_0.65/stability_topN_summary_simulated_dataset_250_0.5_20_0.0.json"
#summary_path = "C:/Users/enthe/Desktop/Thesis/results/stability_predcv_0.65thresh/stability_extended_simulated_dataset_250_0.5_20_0.0.json"
df = pd.read_csv(csv_path)
X = df[[f"X{i}" for i in range(1, 21)]].to_numpy()
y = df["y"].to_numpy()

# --- Load stable indices ---
with open(summary_path) as f:
    summary = json.load(f)

stable_mains = summary["stable_indices"]["mains"]
stable_interactions = summary["stable_indices"]["interactions"]

# --- Build design matrix with main effects and interactions ---
X_stable = X[:, stable_mains] if stable_mains else np.empty((X.shape[0], 0))
inter_terms = []
inter_terms = []
for pair in stable_interactions:
    if len(pair) == 2:
        i, j = pair
        inter_terms.append(X[:, i] * X[:, j])
    elif len(pair) == 1:
        i = pair[0]
        inter_terms.append(X[:, i] ** 2)
if inter_terms:
    X_inter = np.column_stack(inter_terms)
else:
    X_inter = np.empty((X.shape[0], 0))

X_final = np.hstack([X_stable, X_inter])

# --- Fit logistic regression ---
clf = LogisticRegression(max_iter=1000, penalty ='l1', solver = 'liblinear')
clf.fit(X_final, y)

# --- Predict probabilities and plot ROC curve ---
y_prob = clf.predict_proba(X_final)[:, 1]
fpr, tpr, _ = roc_curve(y, y_prob)
roc_auc = roc_auc_score(y, y_prob)
pr_auc = average_precision_score(y, y_prob)
print(f"PR AUC = {pr_auc:.3f}")

plt.figure(figsize=(7, 7))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Stable Main + Interaction Effects)")
plt.legend(loc="lower right")
plt.tight_layout()

# Save ROC curve before showing
stem = os.path.splitext(os.path.basename(csv_path))[0]
out_dir = "C:/Users/enthe/Desktop/Thesis/results/logreg"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, f"roc_curve_{stem}.png")
plt.savefig(out_path)
print(f"ROC curve saved to {out_path}")

plt.show()

precision, recall, _ = precision_recall_curve(y, y_prob)
plt.figure()
plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()

# Save PR curve before showing
pr_out_path = os.path.join(out_dir, f"pr_curve_{stem}.png")
plt.savefig(pr_out_path)
print(f"PR curve saved to {pr_out_path}")


plt.show()


