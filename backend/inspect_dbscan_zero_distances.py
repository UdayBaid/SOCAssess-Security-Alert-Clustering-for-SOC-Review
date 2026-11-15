
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parent.parent
ML_DIR = ROOT / "dataset" / "processed" / "ml"

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

neighbors = NearestNeighbors(
    n_neighbors=6,
    metric="cosine",
    algorithm="brute",
    n_jobs=1
)

neighbors.fit(reduced_X)
distances, neighbor_indices = neighbors.kneighbors(reduced_X)

zero_counts = np.sum(distances[:, 1:] <= 1e-12, axis=1)

print("TOTAL SAMPLED ALERTS:", reduced_X.shape[0])
print("ALERTS WITH AT LEAST 1 ZERO-DISTANCE NEIGHBOR:", int(np.sum(zero_counts >= 1)))
print("ALERTS WITH AT LEAST 5 ZERO-DISTANCE NEIGHBORS:", int(np.sum(zero_counts >= 5)))
print("MAXIMUM ZERO-DISTANCE NEIGHBORS:", int(zero_counts.max()))
print("AVERAGE ZERO-DISTANCE NEIGHBORS:", round(float(zero_counts.mean()), 3))

print("\nZERO-DISTANCE NEIGHBOR COUNT DISTRIBUTION")
print(pd.Series(zero_counts).value_counts().sort_index().to_string())

print("\nSAMPLE ZERO-DISTANCE PAIRS")

shown = 0

for i in range(reduced_X.shape[0]):
    for j in neighbor_indices[i, 1:]:
        if distances[i, list(neighbor_indices[i]).index(j)] <= 1e-12:
            row_a = reduced_X.getrow(i)
            row_b = reduced_X.getrow(j)

            exact_equal = (
                (row_a != row_b).nnz == 0
            )

            print({
                "sample_position_a": int(i),
                "sample_position_b": int(j),
                "original_row_a": int(indices[i]),
                "original_row_b": int(indices[j]),
                "exact_vector_match": exact_equal,
                "nonzero_features_a": int(row_a.nnz),
                "nonzero_features_b": int(row_b.nnz)
            })

            shown += 1
            if shown >= 10:
                break

    if shown >= 10:
        break

print("\nZero-distance inspection complete.")
