
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

X = X[:, keep_indices].tocsr()

seeds = [7, 21, 42, 100]
eps_values = [0.05, 0.08]
sample_size = 5000

results = []

for eps in eps_values:
    assignments = {}
    print(f"\n========== EPSILON {eps} ==========")

    for seed in seeds:
        rng = np.random.default_rng(seed)
        indices = np.sort(
            rng.choice(X.shape[0], size=sample_size, replace=False)
        )

        sample = X[indices]

        model = DBSCAN(
            eps=eps,
            min_samples=5,
            metric="cosine",
            algorithm="brute",
            n_jobs=1
        )

        labels = model.fit_predict(sample)
        assignments[seed] = (indices, labels)

        cluster_sizes = pd.Series(labels).value_counts()
        cluster_count = len(set(labels) - {-1})
        noise_count = int(np.sum(labels == -1))

        sizes = cluster_sizes[cluster_sizes.index != -1]

        result = {
            "eps": eps,
            "seed": seed,
            "clusters": cluster_count,
            "noise_count": noise_count,
            "noise_percentage": round(
                noise_count / sample_size * 100, 2
            ),
            "smallest_cluster": int(sizes.min()) if len(sizes) else 0,
            "largest_cluster": int(sizes.max()) if len(sizes) else 0
        }

        results.append(result)
        print(result)

    print("\nPAIRWISE ARI ON COMMON ALERTS")

    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            seed_a = seeds[i]
            seed_b = seeds[j]

            indices_a, labels_a = assignments[seed_a]
            indices_b, labels_b = assignments[seed_b]

            common, pos_a, pos_b = np.intersect1d(
                indices_a,
                indices_b,
                return_indices=True
            )

            ari = adjusted_rand_score(
                labels_a[pos_a],
                labels_b[pos_b]
            )

            print(
                f"Seeds {seed_a} vs {seed_b}: "
                f"common={len(common)}, ARI={ari:.5f}"
            )

REPORT_DIR.mkdir(parents=True, exist_ok=True)

pd.DataFrame(results).to_csv(
    REPORT_DIR / "dbscan_reduced_stability.csv",
    index=False
)

print("\nDBSCAN reduced-feature stability test complete.")
