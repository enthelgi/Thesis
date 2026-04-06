import pandas as pd
import numpy as np
import json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV


# --- Load normalized TGS dataset ---
csv_path = "data/tgs_data/tgs_dataset_normalized_balanced.csv"
#csv_path = "data/tgs_data/tgs_dataset_normalized.csv"

df = pd.read_csv(csv_path)
X = df.drop(columns=["MCI"]).to_numpy()
y = df["MCI"].to_numpy()



# --- Load selected features/interactions from pliable lasso results ---
#selected_path = "C:/Users/enthe/Desktop/Thesis/results/tgs_results/selected_tgs_imbalanced.json" #nb roc 0.7, pr 0.374 // b roc 0.702, pr 0.365
#selected_path = "C:/Users/enthe/Desktop/Thesis/results/tgs_results/selected_tgs_balanced.json" #b roc 0.630, pr 0.291 // nb 623, 0.289
#selected_path = "C:/Users/enthe/Desktop/Thesis/results/tgs_results/selected_2_tgs_imbalanced.json"
#selected_path = "C:/Users/enthe/Desktop/Thesis/results/tgs_results/selected_2_tgs_2imbalanced.json" #nb 0.610 , 0.268 // b 623, 0.278
#selected_path = "C:/Users/enthe/Desktop/Thesis/results/tgs_results/top10_stability_selected_tgs.json" #b 0.658 0.338// nb 668, 0.373
#selected_path = "C:/Users/enthe/Desktop/Thesis/results/tgs_results/top10_stability_selected_tgs_imbalanced.json" #nb 652, 0.350 // b 0.664, 0.330
#selected_path = "C:/Users/enthe/Desktop/Thesis/results/tgs_results/top10_stability_selected_tgs_balanced.json" # b 0.665, 0.306 // nb 0.662, 0.303
#selected_path = "C:/Users/enthe/Desktop/Thesis/results/tgs_results/top10_stability_selected_tgs_various.json" # nb 0.656 , 0.314 // b 0.671, 0.306

# Tau thresholded selections (balanced)
#selected_path = "C:/Users/enthe/Desktop/Thesis/results/tgs_results/stability_selected_tgs_thresh_0.5_balanced.json" #b 0.708, 0.351 // nb 0.711, 0.373
#selected_path = "C:/Users/enthe/Desktop/Thesis/results/tgs_results/stability_selected_tgs_thresh_0.55_balanced.json" # nb 0.688 , 0.320 // b 0.697 , 0.316 // b 698 0.374 //b 0. 790, 0.798 // nb 789, 0.798
#selected_path = "C:/Users/enthe/Desktop/Thesis/results/tgs_results/stability_selected_tgs_thresh_0.6_balanced.json" # b 0.691, 0.322 // nb 0.683, 0.331
#best
selected_path = "C:/Users/enthe/Desktop/Thesis/results/tgs_results/stability_selected_tgs_thresh_0.65_balanced.json" #nb 0.675, 0.323 // b 0.679, 0.312
#selected_path = "C:/Users/enthe/Desktop/Thesis/results/tgs_results/stability_selected_tgs_thresh_0.7_balanced.json" # b 0.672, 0.306 // nb  0.681, 0.329

# Tau thresholded selections (imbalanced)
# selected_path = "C:/Users/enthe/Desktop/Thesis/results/tgs_results/stability_selected_tgs_thresh_0.5_imbalanced.json"
# selected_path = "C:/Users/enthe/Desktop/Thesis/results/tgs_results/stability_selected_tgs_thresh_0.55_imbalanced.json"
# selected_path = "C:/Users/enthe/Desktop/Thesis/results/tgs_results/stability_selected_tgs_thresh_0.6_imbalanced.json"
# selected_path = "C:/Users/enthe/Desktop/Thesis/results/tgs_results/stability_selected_tgs_thresh_0.65_imbalanced.json"
# selected_path = "C:/Users/enthe/Desktop/Thesis/results/tgs_results/stability_selected_tgs_thresh_0.7_imbalanced.json"

# Various thresholded selections
#selected_path = "C:/Users/enthe/Desktop/Thesis/results/tgs_results/stability_selected_tgs_various_thresh_0.5.json"

with open(selected_path) as f:
    selected = json.load(f)

main_names = selected["selected_mains"]

interactions = selected["selected_interactions"]

# --- Get column indices for selected mains ---
feature_names = df.columns[:-1].tolist()

extra_main = "hsa_miR_X4d_3p"
# if extra_main in feature_names and extra_main not in main_names:
#     main_names.append(extra_main)
main_indices = [feature_names.index(name) for name in main_names if name in feature_names]
# --- Build interaction index set (unordered, no duplicates) ---
interaction_pairs = set()
for inter in interactions:
    i = feature_names.index(inter[0])
    j = feature_names.index(inter[1])
    # i = feature_names.index(inter["main"])
    # j = feature_names.index(inter["modifier"])
    pair = tuple(sorted((i, j)))
    interaction_pairs.add(pair)

# --- Build design matrix ---
X_mains = X[:, main_indices] if main_indices else np.empty((X.shape[0], 0))
inter_terms = []
for i, j in interaction_pairs:
    inter_terms.append(X[:, i] * X[:, j])
X_inter = np.column_stack(inter_terms) if inter_terms else np.empty((X.shape[0], 0))
X_final = np.hstack([X_mains, X_inter])

# --- Fit logistic regression ---
#clf = LogisticRegression(max_iter=1000, class_weight='balanced')
#clf = LogisticRegression(max_iter=1000)
clf = CalibratedClassifierCV(LogisticRegression(max_iter=1000, class_weight='balanced'), method="isotonic")
#Fit L1-penalized logistic regression
# clf = LogisticRegression(
#     penalty='l1',
#     solver='liblinear',  # 'liblinear' supports L1 penalty
#     max_iter=1000,
#     class_weight='balanced'  # optional, use if you want balanced classes
# )
# #clf = CalibratedClassifierCV(LogisticRegression(max_iter=1000), method="isotonic")

clf.fit(X_final, y)

# --- Predict probabilities and plot ROC curve ---
y_prob = clf.predict_proba(X_final)[:, 1]
#fpr, tpr, _ = roc_curve(y, y_prob)
# For ROC curve (Youden's J statistic)
fpr, tpr, thresholds = roc_curve(y, y_prob)
j_scores = tpr - fpr
best_roc_idx = np.argmax(j_scores)
best_roc_threshold = thresholds[best_roc_idx]
roc_auc = roc_auc_score(y, y_prob)


# --- get coefficients from the calibrated logistic regression ---
inner = clf.calibrated_classifiers_[0]
logreg = inner.estimator
coefs = logreg.coef_.ravel()
print("Coefficients:", coefs)
intercept = logreg.intercept_[0]
print("Intercept:", intercept)

# coefs = clf.coef_.ravel()      # shape (n_features,)
# intercept = clf.intercept_[0]  # single intercept for binary
# print("Intercept:", intercept)

feature_labels = [feature_names[i] for i in main_indices] + \
                 [f"{feature_names[i]}*{feature_names[j]}" for i, j in interaction_pairs]

for label, coef in zip(feature_labels, coefs):
    print(f"{label}: {coef:.4f}")


feature_labels = [feature_names[i] for i in main_indices] + \
                 [f"{feature_names[i]}*{feature_names[j]}" for i, j in interaction_pairs]

for label, coef in zip(feature_labels, coefs):
    print(f"{label}: {coef:.4f}")



plt.hist(y_prob, bins=30)
plt.xlabel("Predicted probability")
plt.ylabel("Count")
plt.title("Histogram of predicted probabilities")
plt.show()

# plt.figure(figsize=(7, 7))
# plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
# plt.plot([0, 1], [0, 1], 'k--', lw=1)
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC Curve (Selected Main + Interaction Effects, TGS)")
# plt.legend(loc="lower right")
# plt.tight_layout()
# ROC curve
plt.figure(figsize=(7, 7))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.scatter(fpr[best_roc_idx], tpr[best_roc_idx], color='red', label=f"Best threshold ({best_roc_threshold:.2f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Selected Main + Interaction Effects, TGS)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

plt.show()

# --- PR AUC and Precision-Recall curve ---
#precision, recall, _ = precision_recall_curve(y, y_prob)
# For PR curve (max F1)
precision, recall, pr_thresholds = precision_recall_curve(y, y_prob)
f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
best_pr_idx = np.argmax(f1_scores)
best_pr_threshold = pr_thresholds[best_pr_idx]
pr_auc = average_precision_score(y, y_prob)

# plt.figure(figsize=(7, 7))
# plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.3f}")
# plt.xlabel("Recall")
# plt.ylabel("Precision")
# plt.title("Precision-Recall Curve (Selected Main + Interaction Effects, TGS)")
# plt.legend(loc="lower left")
# plt.tight_layout()
# plt.show()

# PR curve
plt.figure(figsize=(7, 7))
plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.3f}")
plt.scatter(recall[best_pr_idx], precision[best_pr_idx], color='red', label=f"Best threshold ({best_pr_threshold:.2f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (Selected Main + Interaction Effects, TGS)")
plt.legend(loc="lower left")
plt.tight_layout()
plt.show()

print(f"ROC AUC: {roc_auc:.3f}")
print(f"PR AUC: {pr_auc:.3f}")
print(np.bincount(y.astype(int)))
print("Selected mains:", len(main_indices))
print("Selected interactions:", len(interaction_pairs))