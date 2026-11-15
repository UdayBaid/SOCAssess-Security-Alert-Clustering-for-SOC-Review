
import json
import time
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.sparse import load_npz, csr_matrix
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MATRIX_FILE = PROJECT_ROOT / "dataset" / "processed" / "ml" / "alert_features_matrix.npz"
FEATURES_FILE = PROJECT_ROOT / "dataset" / "processed" / "ml" / "feature_names.json"
REPORT_DIR = PROJECT_ROOT / "reports"

FIXED_SIZE = 4000
ADDITIONAL_SIZE = 1000
SAMPLE_SEEDS = [7, 21, 42, 100]
EPS_VALUES = [0.05, 0.08]
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

    print("Loading feature matrix...")

    X = load_npz(MATRIX_FILE).tocsr()

    with open(FEATURES_FILE, "r", encoding="utf-8") as file:
        feature_names = json.load(file)

    if isinstance(feature_names, dict):
        feature_names = feature_names["feature_names"]

    if X.shape[1] != len(feature_names):
        raise ValueError("Feature matrix and feature names do not match.")

    keep_indices = [
        i for i, name in enumerate(feature_names)
        if name not in TEMPORAL_FEATURES
    ]

    X = X[:, keep_indices].tocsr()

    print(f"Alerts available: {X.shape[0]}")
    print(f"Reduced features: {X.shape[1]}")

    if X.shape[0] <= FIXED_SIZE + ADDITIONAL_SIZE:
        raise ValueError("Not enough alerts for the experiment.")

    # Select one fixed group shared by every run.
    rng = np.random.default_rng(42)

    fixed_indices = rng.choice(
        X.shape[0],
        size=FIXED_SIZE,
        replace=False
    )

    fixed_indices = np.sort(fixed_indices)

    fixed_mask = np.ones(X.shape[0], dtype=bool)
    fixed_mask[fixed_indices] = False

    remaining_indices = np.flatnonzero(fixed_mask)

    # Prepare samples: same 4000 alerts + 1000 changing alerts.
    samples = {}

    for seed in SAMPLE_SEEDS:
        sample_rng = np.random.default_rng(seed)

        additional_indices = sample_rng.choice(
            remaining_indices,
            size=ADDITIONAL_SIZE,
            replace=False
        )

        additional_indices = np.sort(additional_indices)

        sample_indices = np.concatenate([
            fixed_indices,
            additional_indices
        ])

        samples[seed] = X[sample_indices]

    print("\nFixed alerts shared across all samples:", FIXED_SIZE)
    print("Additional alerts per sample:", ADDITIONAL_SIZE)

    all_results = []

    for eps in EPS_VALUES:
        print(f"\n========== EPSILON {eps} ==========")

        labels_by_seed = {}

        for seed, sample_matrix in samples.items():
            model = DBSCAN(
                eps=eps,
                min_samples=MIN_SAMPLES,
                metric="cosine",
                algorithm="brute",
                n_jobs=1
            )

            run_start = time.time()
            labels = model.fit_predict(sample_matrix)

            # First 4000 rows are the common fixed alerts.
            fixed_labels = labels[:FIXED_SIZE]

            labels_by_seed[seed] = fixed_labels

            unique_labels, counts = np.unique(
                labels,
                return_counts=True
            )

            cluster_sizes = counts[unique_labels != -1]
            noise_count = int(np.sum(labels == -1))

            result = {
                "eps": eps,
                "seed": seed,
                "clusters": len(cluster_sizes),
                "noise_count": noise_count,
                "noise_percentage": round(
                    noise_count / len(labels) * 100, 2
                ),
                "smallest_cluster": (
                    int(cluster_sizes.min())
                    if len(cluster_sizes) else 0
                ),
                "largest_cluster": (
                    int(cluster_sizes.max())
                    if len(cluster_sizes) else 0
                ),
                "execution_seconds": round(
                    time.time() - run_start, 2
                )
            }

            all_results.append(result)
            print(result)

        print("\nPAIRWISE ARI ON FIXED COMMON ALERTS")

        ari_values = []

        for seed_a, seed_b in combinations(SAMPLE_SEEDS, 2):
            ari = adjusted_rand_score(
                labels_by_seed[seed_a],
                labels_by_seed[seed_b]
            )

            ari_values.append(ari)

            print(
                f"Seeds {seed_a} vs {seed_b}: "
                f"common={FIXED_SIZE}, ARI={ari:.5f}"
            )

        print(f"Average ARI: {np.mean(ari_values):.5f}")
        print(f"Minimum ARI: {np.min(ari_values):.5f}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = REPORT_DIR / "dbscan_controlled_stability.csv"

    pd.DataFrame(all_results).to_csv(
        output_file,
        index=False
    )

    print("\nControlled stability experiment complete.")
    print(f"Report saved at: {output_file}")
    print(f"Total execution time: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
