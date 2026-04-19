'''
takes the TGS stability-selected features/interactions, 
builds a final design matrix, tunes an L1 logistic regression, 
compares it to a dummy baseline and a random forest, 
and makes evaluation/importance plots.
'''

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    precision_recall_curve, average_precision_score, classification_report,
    confusion_matrix, roc_auc_score
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------
# Helpers
# -----------------------
def bootstrap_metric_ci(y_true, y_prob, metric_fn, n_boot=2000, seed=42):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(y_true))
    vals = []
    for _ in range(n_boot):
        b = rng.choice(idx, size=len(idx), replace=True)
        vals.append(metric_fn(y_true[b], y_prob[b]))
    vals = np.array(vals)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), float(vals.mean())


def choose_threshold_by_fbeta(y_true, y_prob, beta=1.0):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    precision_t = precision[:-1]
    recall_t = recall[:-1]
    beta2 = beta**2
    fbeta = (1 + beta2) * (precision_t * recall_t) / (beta2 * precision_t + recall_t + 1e-12)
    best_i = int(np.nanargmax(fbeta))
    return thresholds[best_i], precision_t[best_i], recall_t[best_i], fbeta[best_i]


def plot_prob_overlap(y_true, y_prob, title="Predicted probability distribution by class"):
    plt.figure(figsize=(8, 5))
    df = pd.DataFrame({"y": y_true, "prob": y_prob})
    sns.kdeplot(data=df, x="prob", hue="y", common_norm=False, fill=True, alpha=0.3)
    plt.xlim(0, 1)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_threshold_tradeoff_prob(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    precision_t, recall_t = precision[:-1], recall[:-1]

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, precision_t, label="Precision")
    plt.plot(thresholds, recall_t, label="Recall")
    plt.xlabel("Decision threshold (probability)")
    plt.ylabel("Value")
    plt.title("Precision/Recall vs probability threshold (test)")
    plt.xlim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()


# -----------------------
# 1. Load data
# -----------------------
data = pd.read_csv('data/tgs_data/tgs_dataset_normalized.csv')
stability = pd.read_csv('results/tgs_results/stability_mains_tgs_l1logistic_imbalanced.csv', index_col=0)
inter_stab = pd.read_csv('results/tgs_results/stability_interactions_tgs_l1logistic_imbalanced.csv')

assert 'MCI' in data.columns, "Target column 'MCI' not found."
assert stability.index.is_unique, "Stability index must be unique (covariate names)."

# -----------------------
# 2. Select covariates + interactions
# -----------------------
selected_covariates = stability[stability['freq'] > 0].index.tolist()
missing = [c for c in selected_covariates if c not in data.columns]
if missing:
    raise ValueError(f"Selected covariates missing from dataset: {missing[:10]}{'...' if len(missing)>10 else ''}")

X = data[selected_covariates].copy()

inter_pairs = inter_stab[inter_stab['freq'] > 0][['i', 'j']].values
covariate_names = [c for c in data.columns if c != 'MCI']

for i, j in inter_pairs:
    name_i = covariate_names[i]
    name_j = covariate_names[j]
    if name_i in selected_covariates and name_j in selected_covariates:
        colname = f"{name_i}__X__{name_j}"
        X[colname] = data[name_i] * data[name_j]

y = data['MCI'].astype(int).values
print(f"Positive class prevalence (overall): {y.mean():.3f}  (n={len(y)})")

# -----------------------
# 3. Split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.25, random_state=42
)
print(f"Train prevalence: {y_train.mean():.3f} | Test prevalence: {y_test.mean():.3f}")

# -----------------------
# 4. Baseline
# -----------------------
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
dummy_pred = dummy.predict(X_test)
print("\nDummy (most_frequent) classification report:")
print(classification_report(y_test, dummy_pred, digits=3))
print(f"PR baseline (prevalence) ~ {y_test.mean():.3f}")

# -----------------------
# 5. Tune L1 logistic regression
# -----------------------
param_grid = {'clf__C': np.logspace(-3, 2, 20)}

pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("clf", LogisticRegression(
        penalty='l1',
        solver='saga',
        max_iter=5000,
        class_weight=None,
        random_state=42
    ))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(pipe, param_grid, scoring='average_precision', cv=cv, n_jobs=-1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
print("\nBest parameters:", grid.best_params_)

# -----------------------
# 6. Evaluate using PROBABILITIES
# -----------------------
y_prob = best_model.predict_proba(X_test)[:, 1]
y_prob = best_model.predict_proba(X_test)[:, 1]
precision, recall, _ = precision_recall_curve(y_test, y_prob)
avg_prec = average_precision_score(y_test, y_prob)
print(f"\nAverage Precision (L1 Logistic): {avg_prec:.3f}")

# -----------------------
# 7. Plots (all probability-based)
# -----------------------
# 7a) Probability overlap

plot_prob_overlap(y_test, y_prob, title="Predicted probability distribution by class (test)")

# 7f) Nonlinearity sanity check (RF) - moved up so rf_prob is defined before PR curve plot

rf = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("rf", RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1
    ))
])

rf.fit(X_train, y_train)
rf_prob = rf.predict_proba(X_test)[:, 1]
rf_precision, rf_recall, _ = precision_recall_curve(y_test, rf_prob)
rf_avg_prec = average_precision_score(y_test, rf_prob)
print(f"\nAverage Precision (Random Forest): {rf_avg_prec:.3f}")

# 7b) PR curve

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'L1 Logistic (AP={avg_prec:.2f})')
plt.plot(rf_recall, rf_precision, label=f'Random Forest (AP={rf_avg_prec:.2f})', linestyle='--')
plt.hlines(y_test.mean(), xmin=0, xmax=1, linestyles="--", label=f'Baseline={y_test.mean():.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Test Set)')
plt.legend()
plt.tight_layout()
plt.show()

# 7c) Precision/Recall vs probability threshold  
plot_threshold_tradeoff_prob(y_test, y_prob)

# -----------------------
# 7d) Permutation test + histogram plot 
# -----------------------
rng = np.random.RandomState(42)
perm_aps = []
n_perm = 200

for _ in range(n_perm):
    y_perm = rng.permutation(y_train)
    m = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", LogisticRegression(
            penalty='l1', solver='saga', max_iter=5000,
            C=best_model.named_steps["clf"].C,
            class_weight=None, random_state=42
        ))
    ])
    m.fit(X_train, y_perm)


# -----------------------
# 7e) Permutation importance
# -----------------------
perm_imp = permutation_importance(
    best_model, X_test, y_test,
    scoring="average_precision",
    n_repeats=20,
    random_state=42,
    n_jobs=-1
)
imp = pd.Series(perm_imp.importances_mean, index=X.columns).sort_values(ascending=False)
print("\nTop permutation importances (AP drop):")
print(imp.head(15))

plt.figure(figsize=(10, 6))
topk = imp.head(20)
sns.barplot(x=topk.index, y=topk.values)
plt.xticks(rotation=45, ha='right')
plt.ylabel('Mean AP importance (perm drop)')
plt.title('Top permutation importances (test)')
plt.tight_layout()
plt.show()



# -----------------------
# 8. Nonzero coefficients
# -----------------------
lr = best_model.named_steps["clf"]
coefs = pd.Series(lr.coef_[0], index=X.columns)
nonzero_coefs = coefs[coefs != 0].sort_values(key=np.abs, ascending=False)

print("\nNonzero coefficients (sorted by |coef|):")
print(nonzero_coefs)

plt.figure(figsize=(10, 6))
top_coef = nonzero_coefs.head(25)
sns.barplot(x=top_coef.index, y=top_coef.values)
plt.xticks(rotation=45, ha='right')
plt.ylabel('Coefficient Value')
plt.title('Top nonzero L1 Logistic Regression Coefficients')
plt.tight_layout()
plt.show()
