
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances

ROOT = Path(__file__).resolve().parent.parent
ML_DIR = ROOT / "dataset" / "processed" / "ml"
REPORT_DIR = ROOT / "reports"

X = load_npz(ML_DIR / "alert_features_matrix.npz").tocsr()
metadata = pd.read_csv(ML_DIR / "alert_feature_metadata.csv")
feature_names = pd.read_json(ML_DIR / "feature_names.json", typ="series")

rng = np.random.default_rng(42)
indices = np.sort(rng.choice(X.shape[0], size=5000, replace=False))

sample_X = X[indices]
sample_metadata = metadata.iloc[indices].reset_index(drop=True)

model = DBSCAN(
    eps=0.08,
    min_samples=5,
    metric="cosine",
    algorithm="brute",
    n_jobs=1
)

labels = model.fit_predict(sample_X)

sample_metadata["cluster_id"] = labels

cluster_a = 0
cluster_b = 3

positions_a = np.flatnonzero(labels == cluster_a)
positions_b = np.flatnonzero(labels == cluster_b)

rows = []

for cluster_id, positions in [
    (cluster_a, positions_a),
    (cluster_b, positions_b)
]:
    cluster_matrix = sample_X[positions]

    distances = pairwise_distances(
        cluster_matrix,
        cluster_matrix,
        metric="cosine"
    )

    medoid_local = np.argmin(distances.mean(axis=1))
    medoid_position = positions[medoid_local]

    rows.append({
        "cluster_id": cluster_id,
        "sample_position": int(medoid_position),
        "OrgId": sample_metadata.iloc[medoid_position]["OrgId"],
        "AlertId": sample_metadata.iloc[medoid_position]["AlertId"],
        "average_internal_distance": float(
            distances[medoid_local].mean()
        )
    })

    mean_values = np.asarray(cluster_matrix.mean(axis=0)).ravel()

    print(f"\nCLUSTER {cluster_id}")
    print("Alerts:", len(positions))
    print("Medoid position:", medoid_position)
    print("Medoid OrgId:", sample_metadata.iloc[medoid_position]["OrgId"])
    print("Medoid AlertId:", sample_metadata.iloc[medoid_position]["AlertId"])
    print("Average internal cosine distance:", distances[medoid_local].mean())

    top_indices = np.argsort(mean_values)[-15:][::-1]

    print("\nTOP 15 MEAN FEATURES")
    for idx in top_indices:
        print(feature_names.iloc[idx], round(mean_values[idx], 5))

print("\nMEDOID SUMMARY")
print(pd.DataFrame(rows).to_string(index=False))

REPORT_DIR.mkdir(parents=True, exist_ok=True)
sample_metadata[
    sample_metadata["cluster_id"].isin([cluster_a, cluster_b])
].to_csv(
    REPORT_DIR / "dbscan_clusters_0_3_assignments.csv",
    index=False
)

print("\nComparison complete.")
