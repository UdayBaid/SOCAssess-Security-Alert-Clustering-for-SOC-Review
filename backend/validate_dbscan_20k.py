
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.sparse import load_npz
from sklearn.cluster import DBSCAN


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MATRIX_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "ml"
    / "alert_features_matrix.npz"
)

FEATURES_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "ml"
    / "feature_names.json"
)

METADATA_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "ml"
    / "alert_feature_metadata.csv"
)

REPORT_DIR = PROJECT_ROOT / "reports"
OUTPUT_DIR = PROJECT_ROOT / "dataset" / "processed" / "ml"

SAMPLE_SIZE = 20000
RANDOM_STATE = 42

EPS = 0.05
MIN_SAMPLES = 5

TEMPORAL_FEATURES = {
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "is_weekend"
}


def main():

    start_time = time.time()

    print("=" * 60)
    print("DBSCAN 20,000 ALERT VALIDATION")
    print("=" * 60)

    print("\nLoading feature matrix...")

    X = load_npz(MATRIX_FILE).tocsr()

    with open(FEATURES_FILE, "r", encoding="utf-8") as file:
        feature_names = json.load(file)

    if isinstance(feature_names, dict):
        feature_names = feature_names["feature_names"]

    metadata = pd.read_csv(METADATA_FILE)

    if X.shape[1] != len(feature_names):
        raise ValueError("Feature names do not match matrix.")

    if X.shape[0] != len(metadata):
        raise ValueError("Metadata rows do not match matrix.")

    print("Original matrix:", X.shape)

    keep_indices = [
        i for i, name in enumerate(feature_names)
        if name not in TEMPORAL_FEATURES
    ]

    X = X[:, keep_indices].tocsr()

    print("Reduced matrix:", X.shape)

    if X.shape[0] < SAMPLE_SIZE:
        raise ValueError("Not enough alerts for sampling.")

    print("\nSelecting reproducible sample...")

    rng = np.random.default_rng(RANDOM_STATE)

    sample_indices = np.sort(
        rng.choice(
            X.shape[0],
            size=SAMPLE_SIZE,
            replace=False
        )
    )

    X_sample = X[sample_indices]

    sample_metadata = (
        metadata.iloc[sample_indices]
        .reset_index(drop=True)
    )

    print("Sample shape:", X_sample.shape)

    print("\nTraining DBSCAN...")

    model = DBSCAN(
        eps=EPS,
        min_samples=MIN_SAMPLES,
        metric="cosine",
        algorithm="brute",
        n_jobs=1
    )

    labels = model.fit_predict(X_sample)

    print("\nClustering complete.")

    unique_labels, counts = np.unique(
        labels,
        return_counts=True
    )

    noise_count = int(np.sum(labels == -1))

    cluster_sizes = {
        int(label): int(count)
        for label, count in zip(unique_labels, counts)
        if label != -1
    }

    cluster_count = len(cluster_sizes)

    clustered_count = SAMPLE_SIZE - noise_count

    noise_percentage = (
        noise_count / SAMPLE_SIZE
    ) * 100

    clustered_percentage = (
        clustered_count / SAMPLE_SIZE
    ) * 100

    print("\n" + "=" * 60)
    print("DBSCAN VALIDATION SUMMARY")
    print("=" * 60)

    print("Total alerts:", SAMPLE_SIZE)
    print("Features:", X_sample.shape[1])
    print("Epsilon:", EPS)
    print("Min samples:", MIN_SAMPLES)

    print("Number of clusters:", cluster_count)
    print("Clustered alerts:", clustered_count)
    print("Noise alerts:", noise_count)

    print(f"Noise percentage: {noise_percentage:.2f}%")
    print(f"Clustered percentage: {clustered_percentage:.2f}%")

    if cluster_sizes:
        print("\nCluster size statistics:")

        print("Smallest cluster:", min(cluster_sizes.values()))
        print("Largest cluster:", max(cluster_sizes.values()))

        print(
            "Average cluster size:",
            round(np.mean(list(cluster_sizes.values())), 2)
        )

        print("\nTOP 20 LARGEST CLUSTERS")

        for cluster_id, size in sorted(
            cluster_sizes.items(),
            key=lambda item: item[1],
            reverse=True
        )[:20]:

            percentage = size / SAMPLE_SIZE * 100

            print(
                f"Cluster {cluster_id}: "
                f"{size} alerts ({percentage:.2f}%)"
            )

    print("\nSaving assignments...")

    sample_metadata["cluster_id"] = labels

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    assignments_file = (
        OUTPUT_DIR / "dbscan_20k_assignments.csv"
    )

    sample_metadata.to_csv(
        assignments_file,
        index=False
    )

    report = {
        "algorithm": "DBSCAN",
        "sample_size": SAMPLE_SIZE,
        "random_state": RANDOM_STATE,
        "features": int(X_sample.shape[1]),
        "eps": EPS,
        "min_samples": MIN_SAMPLES,
        "metric": "cosine",
        "cluster_count": cluster_count,
        "clustered_alerts": clustered_count,
        "noise_count": noise_count,
        "noise_percentage": round(noise_percentage, 4),
        "clustered_percentage": round(clustered_percentage, 4),
        "smallest_cluster": (
            min(cluster_sizes.values()) if cluster_sizes else 0
        ),
        "largest_cluster": (
            max(cluster_sizes.values()) if cluster_sizes else 0
        ),
        "cluster_sizes": cluster_sizes,
        "execution_seconds": round(time.time() - start_time, 2)
    }

    report_file = (
        REPORT_DIR / "dbscan_20k_validation.json"
    )

    with open(report_file, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\nAssignments saved:", assignments_file)
    print("Report saved:", report_file)

    print("\nTotal execution time:",
          round(time.time() - start_time, 2), "seconds")

    print("\n20,000-alert DBSCAN validation complete.")


if __name__ == "__main__":
    main()
