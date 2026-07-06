import numpy as np
from drift_detection import population_stability_index, psi_status

baseline = np.random.normal(500, 50, 1000)
new_data = np.random.normal(520, 60, 1000)

psi = population_stability_index(baseline, new_data, bins=10)

print("Population Stability Index (PSI):", round(psi, 4))
print("Status:", psi_status(psi))
