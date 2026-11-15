
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score

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

original_model = DBSCAN(
    eps=0.08,
    min_samples=5,
    metric="cosine",
    algorithm="brute",
    n_jobs=1
)
original_labels = original_model.fit_predict(sample_X)

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

print("Original features:", sample_X.shape[1])
print("Features after removal:", reduced_X.shape[1])
print("Removed features:", sorted(temporal_features))

model = DBSCAN(
    eps=0.08,
    min_samples=5,
    metric="cosine",
    algorithm="brute",
    n_jobs=1
)

new_labels = model.fit_predict(reduced_X)

cluster_sizes = pd.Series(new_labels).value_counts().sort_index()
cluster_count = len(set(new_labels) - {-1})
noise_count = int(np.sum(new_labels == -1))

ari = adjusted_rand_score(original_labels, new_labels)

print("\nDBSCAN WITHOUT TEMPORAL FEATURES")
print("Clusters:", cluster_count)
print("Noise alerts:", noise_count)
print("Noise percentage:", round(noise_count / len(new_labels) * 100, 2))
print("Cluster sizes:")
print(cluster_sizes.to_string())

print("\nCOMPARISON WITH ORIGINAL DBSCAN")
print("Original clusters:", len(set(original_labels) - {-1}))
print("Original noise:", int(np.sum(original_labels == -1)))
print("Adjusted Rand Index:", round(ari, 5))

comparison = pd.DataFrame({
    "original_cluster": original_labels,
    "new_cluster": new_labels
})

print("\nORIGINAL CLUSTERS 0 AND 3: NEW ASSIGNMENTS")
print(
    pd.crosstab(
        comparison.loc[
            comparison["original_cluster"].isin([0, 3]),
            "original_cluster"
        ],
        comparison.loc[
            comparison["original_cluster"].isin([0, 3]),
            "new_cluster"
        ]
    ).to_string()
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)
comparison.to_csv(
    REPORT_DIR / "dbscan_temporal_comparison.csv",
    index=False
)

print("\nTemporal feature experiment complete.")
