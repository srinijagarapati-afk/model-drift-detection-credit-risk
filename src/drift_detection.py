import numpy as np
from scipy.stats import ks_2samp


def population_stability_index(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Calculate PSI between a baseline (expected) distribution and a new (actual) distribution.

    PSI interpretation:
    - < 0.1   : no significant drift
    - 0.1-0.25: moderate drift
    - > 0.25  : significant drift
    """
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


def kolmogorov_smirnov_test(expected, actual):
    """
    KS test to detect distribution difference
    """
    stat, p_value = ks_2samp(expected, actual)
    return stat, p_value


def psi_status(psi):
    if psi < 0.1:
        return "No significant drift"
    elif psi < 0.25:
        return "Moderate drift"
    else:
        return "Significant drift detected"
