"""
Model Drift Detection for Credit Risk Models
Detects data drift and prediction drift using statistical tests.
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def population_stability_index(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Calculate PSI between a baseline (expected) distribution and a
    new (actual) distribution. PSI < 0.1 = no significant drift,
    0.1-0.25 = moderate drift, > 0.25 = significant drift.
    """
    breakpoints = np.linspace(0, 100, bins + 1)
    bucket_edges = np.percentile(expected, breakpoints)
    bucket_edges[0] = -np.inf
    bucket_edges[-1] = np.inf

    expected_percents = np.histogram(expected, bucket_edges)[0] / len(expected)
    actual_percents = np.histogram(actual, bucket_edges)[0] / len(actual)

    # Avoid division by zero
    expected_percents = np.where(expected_percents == 0, 1e-4, expected_percents)
    actual_percents = np.where(actual_percents == 0, 1e-4, actual_percents)

    psi = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
    return psi


def ks_test_drift(expected: np.ndarray, actual: np.ndarray, alpha: float = 0.05) -> dict:
    """
    Run a Kolmogorov-Smirnov test to check if two samples come from
    the same distribution. Returns statistic, p-value, and a drift flag.
    """
    statistic, p_value = ks_2samp(expected, actual)
    return {
        "statistic": statistic,
        "p_value": p_value,
        "drift_detected": p_value < alpha,
    }


def check_feature_drift(baseline_df: pd.DataFrame, current_df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    Run PSI and KS test across a list of features and return a summary table.
    """
    results = []
    for feature in features:
        psi = population_stability_index(baseline_df[feature].values, current_df[feature].values)
        ks = ks_test_drift(baseline_df[feature].values, current_df[feature].values)
        results.append({
            "feature": feature,
            "psi": psi,
            "psi_flag": "significant" if psi > 0.25 else ("moderate" if psi > 0.1 else "none"),
            "ks_statistic": ks["statistic"],
            "ks_p_value": ks["p_value"],
            "ks_drift_detected": ks["drift_detected"],
        })
    return pd.DataFrame(results)


if __name__ == "__main__":
    # Example usage with synthetic data
    np.random.seed(42)
    baseline = pd.DataFrame({
        "credit_score": np.random.normal(650, 50, 1000),
        "debt_to_income": np.random.normal(0.3, 0.1, 1000),
    })
    current = pd.DataFrame({
        "credit_score": np.random.normal(620, 55, 1000),  # shifted
        "debt_to_income": np.random.normal(0.3, 0.1, 1000),  # unchanged
    })

    report = check_feature_drift(baseline, current, ["credit_score", "debt_to_income"])
    print(report)
