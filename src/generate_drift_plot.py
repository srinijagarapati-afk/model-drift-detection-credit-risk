import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from drift_detection import population_stability_index, ks_test_drift

# Load your actual data
baseline = pd.read_csv("../data/baseline_period.csv")
current = pd.read_csv("../data/current_period.csv")

# Pick the first numeric column to visualize (change this to a real feature name if you know one)
feature = baseline.select_dtypes(include=[np.number]).columns[0]

baseline_values = baseline[feature].values
current_values = current[feature].values

psi = population_stability_index(baseline_values, current_values)
ks_result = ks_test_drift(baseline_values, current_values)

# Create the plot
plt.figure(figsize=(10, 6))
plt.hist(baseline_values, bins=30, alpha=0.6, label="Baseline Period", color="#4C72B0", density=True)
plt.hist(current_values, bins=30, alpha=0.6, label="Current Period", color="#DD8452", density=True)
plt.title(f"Distribution Drift for '{feature}'\nPSI = {psi:.4f} | KS p-value = {ks_result['p_value']:.4f}")
plt.xlabel(feature)
plt.ylabel("Density")
plt.legend()
plt.tight_layout()

os.makedirs("../assets", exist_ok=True)
plt.savefig("../assets/drift_plot.png", dpi=150)
print(f"Saved plot for feature: {feature}")
print(f"PSI: {psi:.4f}")
print(f"KS drift detected: {ks_result['drift_detected']}")
