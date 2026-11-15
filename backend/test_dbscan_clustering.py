
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parent.parent
ML_DIR = ROOT / "dataset" / "processed" / "ml"
REPORT_DIR = ROOT / "reports"

X = load_npz(ML_DIR / "alert_features_matrix.npz").tocsr()

baseline = pd.read_csv(ML_DIR / "kmeans_k3_assignments.csv")
baseline_labels = baseline["cluster_id"].to_numpy()

SAMPLE_SIZE = 5000
RANDOM_STATE = 42

rng = np.random.default_rng(RANDOM_STATE)
indices = np.sort(
    rng.choice(X.shape[0], size=SAMPLE_SIZE, replace=False)
)

X_sample = X[indices]
kmeans_sample_labels = baseline_labels[indices]

configs = [
    (0.03, 5),
    (0.08, 5),
    (0.15, 5),
    (0.08, 10)
]

results = []

for eps, min_samples in configs:
    print(
        f"\nTesting eps={eps}, min_samples={min_samples}",
        flush=True
    )

    start = time.time()

    model = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="cosine",
        algorithm="brute",
        n_jobs=1
    )

    labels = model.fit_predict(X_sample)

    unique, counts = np.unique(labels, return_counts=True)

    cluster_sizes = {
        str(int(cluster)): int(count)
        for cluster, count in zip(unique, counts)
    }

    noise_count = int(np.sum(labels == -1))
    cluster_labels = labels[labels != -1]
    cluster_count = len(set(cluster_labels))

    non_noise_sizes = [
        int(count)
        for cluster, count in zip(unique, counts)
        if cluster != -1
    ]

    ari = adjusted_rand_score(
        kmeans_sample_labels,
        labels
    )

    result = {
        "eps": eps,
        "min_samples": min_samples,
        "cluster_count": cluster_count,
        "noise_count": noise_count,
        "noise_percentage": round(
            noise_count / SAMPLE_SIZE * 100, 2
        ),
        "largest_cluster": max(non_noise_sizes, default=0),
        "smallest_cluster": min(non_noise_sizes, default=0),
        "cluster_sizes": cluster_sizes,
        "ari_vs_kmeans": float(ari),
        "execution_seconds": round(time.time() - start, 2)
    }

    results.append(result)

    print("Clusters:", cluster_count)
    print("Noise:", noise_count, f"({result['noise_percentage']}%)")
    print("Largest cluster:", result["largest_cluster"])
    print("ARI vs K-Means:", round(ari, 4))
    print("Time:", result["execution_seconds"], "seconds")

REPORT_DIR.mkdir(parents=True, exist_ok=True)

with open(REPORT_DIR / "dbscan_clustering_results.json", "w") as file:
    json.dump(results, file, indent=2)

print("\nDBSCAN clustering experiment complete.")
