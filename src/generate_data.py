"""
Generates synthetic credit risk data for demonstrating drift detection.
Creates a baseline period and a current period with intentional drift
in some features.
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

n = 2000

# Baseline period (e.g., last quarter's applicants)
baseline = pd.DataFrame({
    "credit_score": np.random.normal(650, 50, n),
    "debt_to_income": np.random.normal(0.30, 0.08, n),
    "annual_income": np.random.normal(60000, 15000, n),
    "credit_utilization": np.random.beta(2, 5, n),
    "num_open_accounts": np.random.poisson(4, n),
})

# Current period (e.g., this quarter's applicants) — drift introduced
current = pd.DataFrame({
    "credit_score": np.random.normal(615, 60, n),          # drifted: lower, wider
    "debt_to_income": np.random.normal(0.30, 0.08, n),      # unchanged
    "annual_income": np.random.normal(58000, 16000, n),     # slight drift
    "credit_utilization": np.random.beta(3, 4, n),          # drifted: shifted higher
    "num_open_accounts": np.random.poisson(4, n),           # unchanged
})

os.makedirs("data", exist_ok=True)
baseline.to_csv("data/baseline_period.csv", index=False)
current.to_csv("data/current_period.csv", index=False)

print(f"Saved baseline_period.csv ({len(baseline)} rows)")
print(f"Saved current_period.csv ({len(current)} rows)")
