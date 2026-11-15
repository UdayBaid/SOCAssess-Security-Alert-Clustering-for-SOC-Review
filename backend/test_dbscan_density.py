
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

configs = [
    (0.03, 5),
    (0.08, 5),
    (0.15, 5),
    (0.20, 5),
    (0.08, 10)
]

results = []

for eps, min_samples in configs:
    print(f"\nTesting eps={eps}, min_samples={min_samples}", flush=True)

    start = time.time()

    neighbors = NearestNeighbors(
        radius=eps,
        metric="cosine",
        algorithm="brute",
        n_jobs=1
    )

    neighbors.fit(X_sample)

    neighborhoods = neighbors.radius_neighbors(
        X_sample,
        return_distance=False
    )

    counts = np.array([len(x) for x in neighborhoods])

    core_count = int(np.sum(counts >= min_samples))
    noise_candidate_count = int(np.sum(counts < min_samples))

    result = {
        "eps": eps,
        "min_samples": min_samples,
        "mean_neighbors": float(np.mean(counts)),
        "median_neighbors": float(np.median(counts)),
        "core_point_candidates": core_count,
        "non_core_candidates": noise_candidate_count,
        "execution_seconds": round(time.time() - start, 2)
    }

    results.append(result)

    print("Mean neighbors:", round(result["mean_neighbors"], 2))
    print("Median neighbors:", result["median_neighbors"])
    print("Core point candidates:", core_count)
    print("Non-core candidates:", noise_candidate_count)
    print("Time:", result["execution_seconds"])

REPORT_DIR.mkdir(parents=True, exist_ok=True)

with open(REPORT_DIR / "dbscan_density_test.json", "w") as file:
    json.dump(results, file, indent=2)

print("\nDBSCAN density profiling complete.")

