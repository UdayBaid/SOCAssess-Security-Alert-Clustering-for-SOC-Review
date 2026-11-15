
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.cluster import DBSCAN

ROOT = Path(__file__).resolve().parent.parent
ML_DIR = ROOT / "dataset" / "processed" / "ml"
REPORT_DIR = ROOT / "reports"

X = load_npz(ML_DIR / "alert_features_matrix.npz").tocsr()
metadata = pd.read_csv(ML_DIR / "alert_feature_metadata.csv")
feature_names = pd.read_json(
    ML_DIR / "feature_names.json",
    typ="series"
).tolist()

rng = np.random.default_rng(42)
indices = np.sort(rng.choice(X.shape[0], size=5000, replace=False))

sample_X = X[indices]
model = DBSCAN(
    eps=0.08,
    min_samples=5,
    metric="cosine",
    algorithm="brute",
    n_jobs=1
)

labels = model.fit_predict(sample_X)

positions_0 = np.flatnonzero(labels == 0)
positions_3 = np.flatnonzero(labels == 3)

mean_0 = np.asarray(
    sample_X[positions_0].mean(axis=0)
).ravel()

mean_3 = np.asarray(
    sample_X[positions_3].mean(axis=0)
).ravel()

difference = mean_3 - mean_0

comparison = pd.DataFrame({
    "feature": feature_names,
    "cluster_0_mean": mean_0,
    "cluster_3_mean": mean_3,
    "difference_3_minus_0": difference,
    "absolute_difference": np.abs(difference)
})

comparison = comparison.sort_values(
    "absolute_difference",
    ascending=False
)

print("\nTOP FEATURES DIFFERENTIATING CLUSTER 0 AND 3")
print(comparison.head(25).to_string(index=False))


print("\nWEEKEND DISTRIBUTION")

weekend_index = feature_names.index("is_weekend")
weekend_values = np.asarray(
    sample_X[:, weekend_index].toarray()
).ravel()

for cluster_id in [0, 3]:
    mask = labels == cluster_id
    values = weekend_values[mask]

    print(
        f"Cluster {cluster_id}:",
        {
            "weekend_0": int(np.sum(values == 0)),
            "weekend_1": int(np.sum(values == 1)),
            "total": len(values)
        }
    )


REPORT_DIR.mkdir(parents=True, exist_ok=True)
comparison.to_csv(
    REPORT_DIR / "dbscan_cluster_0_3_feature_differences.csv",
    index=False
)

print("\nFeature difference analysis complete.")
