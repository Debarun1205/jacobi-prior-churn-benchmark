import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score
)

RANDOM_STATE = 42

# ---- Load data exactly as the original notebook did ----
df = pd.read_csv('data_for_predictions.csv')
drop_cols = ['Unnamed: 0', 'id']
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

X = df.drop(columns=['churn'])
y = df['churn'].astype(float).values
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}, churn rate train={y_train.mean():.4f} test={y_test.mean():.4f}")

# =========================================================
# 1. Jacobian / Jacobi-Prior closed-form Binomial-logit estimator
#    (Das & Dey 2006, Table 1, flat prior a=b=1, m_i=1 -> Bernoulli):
#    beta_hat = (X'X)^-1 X' log((y+1)/(2-y))
# =========================================================

# Standardize features (not in the original 2006 paper, but necessary here:
# the 68-feature engineered set is highly collinear/differently scaled,
# so raw (X'X) is near-singular -- standardizing is a reasonable, disclosed adaptation)
scaler = StandardScaler()
Xtr_s = scaler.fit_transform(X_train)
Xte_s = scaler.transform(X_test)

# add intercept column
Xtr_design = np.column_stack([np.ones(len(Xtr_s)), Xtr_s])
Xte_design = np.column_stack([np.ones(len(Xte_s)), Xte_s])

# closed-form target transform (flat prior a=1, b=1, Bernoulli m_i=1)
z_train = np.log((y_train + 1) / (2 - y_train))

t0 = time.perf_counter()
XtX = Xtr_design.T @ Xtr_design
XtZ = Xtr_design.T @ z_train
beta_hat = np.linalg.pinv(XtX) @ XtZ   # pinv used since (X'X) is near-singular; equivalent to Das & Dey's (X'X)^-1 when full rank
jacobi_fit_time = time.perf_counter() - t0

# posterior-mode log-odds -> probability via inverse logit
eta_test = Xte_design @ beta_hat
p_jacobi = 1 / (1 + np.exp(-eta_test))

t0 = time.perf_counter()
eta_test_for_timing = Xte_design @ beta_hat  # inference cost
jacobi_infer_time = time.perf_counter() - t0

jacobi_metrics = {
    'ROC-AUC': roc_auc_score(y_test, p_jacobi),
    'PR-AUC': average_precision_score(y_test, p_jacobi),
    'Accuracy@0.5': accuracy_score(y_test, (p_jacobi >= 0.5).astype(int)),
    'Precision@0.5': precision_score(y_test, (p_jacobi >= 0.5).astype(int), zero_division=0),
    'Recall@0.5': recall_score(y_test, (p_jacobi >= 0.5).astype(int), zero_division=0),
    'F1@0.5': f1_score(y_test, (p_jacobi >= 0.5).astype(int), zero_division=0),
    'Fit time (s)': jacobi_fit_time,
    'Inference time (s)': jacobi_infer_time,
}

# =========================================================
# 2. Random Forest baseline (re-fit here, same hyperparams as the
#    original notebook, so runtime comparison is measured in the
#    same environment)
# =========================================================
rf = RandomForestClassifier(
    n_estimators=1000, max_depth=None, min_samples_leaf=5,
    max_features='sqrt', class_weight='balanced',
    random_state=RANDOM_STATE, n_jobs=-1
)

t0 = time.perf_counter()
rf.fit(X_train, y_train)
rf_fit_time = time.perf_counter() - t0

t0 = time.perf_counter()
p_rf = rf.predict_proba(X_test)[:, 1]
rf_infer_time = time.perf_counter() - t0

y_pred_rf = (p_rf >= 0.5).astype(int)
rf_metrics = {
    'ROC-AUC': roc_auc_score(y_test, p_rf),
    'PR-AUC': average_precision_score(y_test, p_rf),
    'Accuracy@0.5': accuracy_score(y_test, y_pred_rf),
    'Precision@0.5': precision_score(y_test, y_pred_rf, zero_division=0),
    'Recall@0.5': recall_score(y_test, y_pred_rf, zero_division=0),
    'F1@0.5': f1_score(y_test, y_pred_rf, zero_division=0),
    'Fit time (s)': rf_fit_time,
    'Inference time (s)': rf_infer_time,
}

results = pd.DataFrame({'Jacobian/Jacobi-Prior (closed-form)': jacobi_metrics, 'Random Forest (baseline)': rf_metrics})
print()
print(results.round(4).to_string())
results.round(4).to_csv('jacobi_vs_rf_results.csv')
print("\nSaved: jacobi_vs_rf_results.csv")
