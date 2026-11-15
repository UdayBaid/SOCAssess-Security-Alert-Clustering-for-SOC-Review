
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parent.parent
ML_DIR = ROOT / "dataset" / "processed" / "ml"
REPORT_DIR = ROOT / "reports"

X = load_npz(ML_DIR / "alert_features_matrix.npz").tocsr()

feature_names = pd.read_json(
    ML_DIR / "feature_names.json",
    typ="series"
).tolist()

rng = np.random.default_rng(42)
indices = np.sort(rng.choice(X.shape[0], size=5000, replace=False))

sample_X = X[indices]

temporal_features = {
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "is_weekend"
}

keep_indices = [
    i for i, name in enumerate(feature_names)
    if name not in temporal_features
]

reduced_X = sample_X[:, keep_indices].tocsr()

def profile_distances(matrix, name):
    neighbors = NearestNeighbors(
        n_neighbors=6,
        metric="cosine",
        algorithm="brute",
        n_jobs=1
    )

    neighbors.fit(matrix)
    distances, _ = neighbors.kneighbors(matrix)

    fifth_neighbor = distances[:, 5]

    percentiles = [25, 50, 75, 90, 95, 99]

    print(f"\n{name}")
    print("5th-neighbor distance percentiles")

    for percentile in percentiles:
        value = np.percentile(fifth_neighbor, percentile)
        print(f"{percentile}th percentile: {value:.5f}")

    return {
        "feature_set": name,
        **{
            f"p{p}": float(np.percentile(fifth_neighbor, p))
            for p in percentiles
        }
    }

results = [
    profile_distances(sample_X, "original_433_features"),
    profile_distances(reduced_X, "without_temporal_428_features")
]

REPORT_DIR.mkdir(parents=True, exist_ok=True)

pd.DataFrame(results).to_csv(
    REPORT_DIR / "dbscan_distance_profile_comparison.csv",
    index=False
)

print("\nDistance profile comparison complete.")
