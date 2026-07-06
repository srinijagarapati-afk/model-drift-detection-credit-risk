"""
Unit tests for drift detection functions.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from drift_detection import population_stability_index, ks_test_drift, check_feature_drift
import pandas as pd


def test_psi_no_drift():
    """Identical distributions should have PSI near 0."""
    np.random.seed(0)
    data = np.random.normal(0, 1, 5000)
    psi = population_stability_index(data, data.copy())
    assert psi < 0.01, f"Expected near-zero PSI for identical data, got {psi}"


def test_psi_detects_drift():
    """A clearly shifted distribution should have high PSI."""
    np.random.seed(0)
    baseline = np.random.normal(0, 1, 5000)
    shifted = np.random.normal(5, 1, 5000)
    psi = population_stability_index(baseline, shifted)
    assert psi > 0.25, f"Expected significant PSI for shifted data, got {psi}"


def test_ks_no_drift():
    """Identical distributions should not trigger drift detection."""
    np.random.seed(1)
    data = np.random.normal(0, 1, 2000)
    result = ks_test_drift(data, data.copy())
    assert result["drift_detected"] is False


def test_ks_detects_drift():
    """A clearly shifted distribution should trigger drift detection."""
    np.random.seed(1)
    baseline = np.random.normal(0, 1, 2000)
    shifted = np.random.normal(3, 1, 2000)
    result = ks_test_drift(baseline, shifted)
    assert result["drift_detected"] is True


def test_check_feature_drift_output_shape():
    """check_feature_drift should return one row per feature with expected columns."""
    np.random.seed(2)
    baseline_df = pd.DataFrame({
        "feature_a": np.random.normal(0, 1, 1000),
        "feature_b": np.random.normal(0, 1, 1000),
    })
    current_df = pd.DataFrame({
        "feature_a": np.random.normal(0, 1, 1000),
        "feature_b": np.random.normal(5, 1, 1000),
    })

    report = check_feature_drift(baseline_df, current_df, ["feature_a", "feature_b"])

    assert len(report) == 2
    expected_columns = {"feature", "psi", "psi_flag", "ks_statistic", "ks_p_value", "ks_drift_detected"}
    assert expected_columns.issubset(set(report.columns))
