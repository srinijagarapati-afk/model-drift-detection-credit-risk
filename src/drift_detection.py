import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def population_stability_index(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    breakpoints = np.linspace(0, 100, bins + 1)
    bucket_edges = np.percentile(expected, breakpoints)
    bucket_edges[0] = -np.inf
    bucket_edges[-1] = np.inf

    expected_counts = np.histogram(expected, bucket_edges)[0]
    actual_counts = np.histogram(actual, bucket_edges)[0]

    expected_perc = expected_counts / len(expected)
    actual_perc = actual_counts / len(actual)

    expected_perc = np.where(expected_perc == 0, 1e-6, expected_perc)
    actual_perc = np.where(actual_perc == 0, 1e-6, actual_perc)

    psi = np.sum((expected_perc - actual_perc) * np.log(expected_perc / actual_perc))
    return psi


def psi_status(psi: float) -> str:
    if psi < 0.1:
        return "No significant drift"
    elif psi < 0.25:
        return "Moderate drift"
    else:
        return "Significant drift detected"


def ks_test_drift(baseline: np.ndarray, current: np.ndarray, alpha: float = 0.05) -> dict:
    stat, p_value = ks_2samp(baseline, current)
    return {
        "ks_statistic": stat,
        "p_value": p_value,
        "drift_detected": bool(p_value < alpha),
    }


def check_feature_drift(baseline_df: pd.DataFrame, current_df: pd.DataFrame, features: list, psi_threshold: float = 0.25, alpha: float = 0.05) -> pd.DataFrame:
    rows = []
    for feature in features:
        baseline_values = baseline_df[feature].values
        current_values = current_df[feature].values

        psi = population_stability_index(baseline_values, current_values)
        ks_result = ks_test_drift(baseline_values, current_values, alpha=alpha)

        rows.append({
            "feature": feature,
            "psi": psi,
            "psi_flag": psi > psi_threshold,
            "ks_statistic": ks_result["ks_statistic"],
            "ks_p_value": ks_result["p_value"],
            "ks_drift_detected": ks_result["drift_detected"],
        })

    return pd.DataFrame(rows)
