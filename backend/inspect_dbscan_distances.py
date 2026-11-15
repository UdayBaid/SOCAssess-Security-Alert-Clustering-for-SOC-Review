
from pathlib import Path
import json
import time

import numpy as np
from scipy.sparse import load_npz
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parent.parent
ML_DIR = ROOT / "dataset" / "processed" / "ml"
REPORT_DIR = ROOT / "reports"

X = load_npz(ML_DIR / "alert_features_matrix.npz").tocsr()

SAMPLE_SIZE = 5000
RANDOM_STATE = 42

rng = np.random.default_rng(RANDOM_STATE)
indices = np.sort(
    rng.choice(X.shape[0], size=SAMPLE_SIZE, replace=False)
)

X_sample = X[indices]

print("Sample shape:", X_sample.shape)
print("Calculating nearest-neighbor distances...", flush=True)

start = time.time()

neighbors = NearestNeighbors(
    n_neighbors=5,
    metric="cosine",
    algorithm="brute",
    n_jobs=1
)

neighbors.fit(X_sample)

distances, _ = neighbors.kneighbors(X_sample)

fifth_neighbor_distances = distances[:, -1]

quantiles = {
    str(q): float(np.quantile(fifth_neighbor_distances, q / 100))
    for q in [25, 50, 75, 90, 95, 99]
}

report = {
    "sample_size": SAMPLE_SIZE,
    "features": int(X_sample.shape[1]),
    "metric": "cosine",
    "min_samples_reference": 5,
    "distance_quantiles": quantiles,
    "execution_seconds": round(time.time() - start, 2)
}

REPORT_DIR.mkdir(parents=True, exist_ok=True)

with open(REPORT_DIR / "dbscan_distance_profile.json", "w") as file:
    json.dump(report, file, indent=2)

print("\n5th-neighbor distance quantiles:")
for q, value in quantiles.items():
    print(f"{q}th percentile: {value:.4f}")

print("\nExecution time:", report["execution_seconds"], "seconds")
print("Report:", REPORT_DIR / "dbscan_distance_profile.json")
