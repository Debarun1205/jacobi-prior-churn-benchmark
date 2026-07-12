Jacobi Prior / Jacobian Closed-Form Estimator — Churn Prediction Benchmark
Benchmarking the closed-form Binomial-logit estimator from Das & Dey (2006), "On Bayesian
Analysis of Generalized Linear Models Using the Jacobian Technique" (The American Statistician),
and the Jacobi Prior framework from Das & Sardar (2024),
"Jacobi Prior: An Alternative Bayesian Method for Supervised Learning",
against a Random Forest baseline on a real applied dataset: SME customer churn prediction
(PowerCo dataset, BCG X Data Science Simulation).
This repo was put together as a follow-up to a discussion with Prof. Sourish Das (Chennai
Mathematical Institute) — reference implementation: https://github.com/sourish-cmi/Jacobi-Prior/
Data
The churn dataset (`data_for_predictions.csv`, 14,606 SME customers, 61 engineered features,
9.7% churn rate) is from an earlier, separate project and is available here:
https://github.com/Debarun1205/BCGX-Exploratory-data-analysis
It is not duplicated in this repo — `jacobi_benchmark.py` expects `data_for_predictions.csv`
in the working directory (download it from the link above).
Method
Estimator implemented: flat-prior (a = b = 1) Binomial-logit closed form from Das & Dey
(2006), Table 1:
```
  beta_hat = (X'X)^-1 X' log((y + 1) / (2 - y))
  ```
applied to the Bernoulli churn indicator.
Baseline: the original Random Forest configuration from the churn project
(1000 trees, `class_weight='balanced'`, `max_features='sqrt'`), re-fit in the same
environment for a fair runtime comparison.
Split: identical 75/25 stratified train/test split (`random_state=42`) used by the
original Random Forest notebook.
Disclosed deviations from the literal formula
Standardization: features were standardized (zero mean, unit variance) before applying
the estimator. This is not part of the original derivation, but was necessary here — the
61-feature engineered set is collinear enough that the raw `(X'X)` matrix is close to singular.
Pseudoinverse: due to that near-singularity, the Moore-Penrose pseudoinverse
(`np.linalg.pinv`) was used in place of a literal matrix inverse. This is the standard,
minimum-norm generalization, and is exact when `(X'X)` is full rank.
Results
Metric	Jacobian / Jacobi-Prior (closed-form)	Random Forest (baseline)
ROC-AUC	0.650	0.707
PR-AUC	0.175	0.305
Precision / Recall / F1 (top-9.7% risk threshold)	0.231 / 0.231 / 0.231	0.327 / 0.327 / 0.327
Fit time	0.009 sec	21.52 sec
Inference time	0.0001 sec	0.47 sec
Full discussion in `results/Jacobi_Prior_Churn_Benchmark_Report.pdf`.
Summary: the closed-form estimator reproduces the paper's central speed claim clearly
(~2,300x faster to fit, ~4,700x faster at inference), but Random Forest is meaningfully more
accurate on this dataset (higher ROC-AUC, PR-AUC, and top-risk precision/recall). This is a
real, disclosed limitation rather than a tuned success story — the estimator is linear in the
transformed log-odds, while this dataset's 61 correlated, engineered features appear to reward
a nonlinear model more than the cleaner feature spaces (SDSS classification, spinal MRI
features) used in the original paper's real-data experiments.
Reproducing
```bash
pip install pandas numpy scikit-learn
# download data_for_predictions.csv from the BCGX-Exploratory-data-analysis repo into this folder
python3 jacobi_benchmark.py
```
Contents
```
jacobi-prior-churn-benchmark/
├── jacobi_benchmark.py                        # estimator implementation + benchmark script
├── results/
│   ├── jacobi_vs_rf_results.csv               # raw output metrics
│   └── Jacobi_Prior_Churn_Benchmark_Report.pdf # full write-up
└── README.md
```
