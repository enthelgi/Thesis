import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# --- Load selection info ---
with open("C:/Users/enthe/Desktop/Thesis/results/tgs_results/selected_tgs_balanced.json", "r") as f:
    sel = json.load(f)

selected_mains_pl = sel["selected_mains_pl"]
dataset_path = "C:/Users/enthe/Desktop/Thesis/data/tgs_data/tgs_dataset_normalized_balanced.csv"

# --- Load dataset ---
df = pd.read_csv(dataset_path)
# Assume last column is target
X = df[selected_mains_pl].to_numpy(dtype=float)
y = df.iloc[:, -1].to_numpy()

# --- Fit logistic regression (L1 penalty) ---
clf = LogisticRegression(
    penalty='l1',
    solver='liblinear',
    max_iter=1000,
    class_weight='balanced'
)
clf.fit(X, y)
y_pred = clf.predict(X)
accuracy = np.mean(y == y_pred)
coefs = dict(zip(selected_mains_pl, clf.coef_[0].tolist()))

print("Selected features:", selected_mains_pl)
print("Coefficients:", coefs)
print("Intercept:", clf.intercept_[0])
print("Accuracy:", accuracy)

# ...existing code...
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# --- Predict probabilities and plot ROC curve ---
y_prob = clf.predict_proba(X)[:, 1]
auc = roc_auc_score(y, y_prob)
fpr, tpr, _ = roc_curve(y, y_prob)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Logistic Regression on Selected Mains)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

print("AUC:", auc)
# ...existing code...