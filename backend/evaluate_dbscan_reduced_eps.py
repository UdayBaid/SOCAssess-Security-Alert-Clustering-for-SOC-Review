
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

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

eps_values = [0.001, 0.005, 0.01, 0.02, 0.05, 0.08, 0.15]
results = []

for eps in eps_values:
    model = DBSCAN(
        eps=eps,
        min_samples=5,
        metric="cosine",
        algorithm="brute",
        n_jobs=1
    )

    labels = model.fit_predict(reduced_X)

    cluster_ids = set(labels) - {-1}
    cluster_count = len(cluster_ids)
    noise_count = int(np.sum(labels == -1))

    valid_mask = labels != -1
    valid_labels = labels[valid_mask]

    silhouette = None

    if (
        len(set(valid_labels)) >= 2
        and len(valid_labels) > len(set(valid_labels))
    ):
        silhouette = silhouette_score(
            reduced_X[valid_mask],
            valid_labels,
            metric="cosine",
            sample_size=min(2000, int(valid_mask.sum())),
            random_state=42
        )

    cluster_sizes = pd.Series(labels).value_counts()

    result = {
        "eps": eps,
        "clusters": cluster_count,
        "noise_count": noise_count,
        "noise_percentage": round(noise_count / len(labels) * 100, 2),
        "largest_cluster": int(cluster_sizes[cluster_sizes.index != -1].max())
        if cluster_count else 0,
        "smallest_cluster": int(cluster_sizes[cluster_sizes.index != -1].min())
        if cluster_count else 0,
        "silhouette_non_noise": silhouette
    }

    results.append(result)

    print("\nEPS:", eps)
    for key, value in result.items():
        if key != "eps":
            print(f"{key}: {value}")

REPORT_DIR.mkdir(parents=True, exist_ok=True)

pd.DataFrame(results).to_csv(
    REPORT_DIR / "dbscan_reduced_eps_comparison.csv",
    index=False
)

print("\nReduced-feature DBSCAN evaluation complete.")
