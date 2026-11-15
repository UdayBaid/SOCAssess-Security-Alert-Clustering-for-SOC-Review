
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import load_npz

ROOT = Path(__file__).resolve().parent.parent
ML_DIR = ROOT / "dataset" / "processed" / "ml"

X = load_npz(ML_DIR / "alert_features_matrix.npz").tocsr()
metadata = pd.read_csv(ML_DIR / "alert_feature_metadata.csv")

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

row_a = 329
row_b = 76319

for row_id in [row_a, row_b]:
    row = X.getrow(row_id)

    print(f"\nORIGINAL ROW: {row_id}")
    print("OrgId:", metadata.iloc[row_id]["OrgId"])
    print("AlertId:", metadata.iloc[row_id]["AlertId"])
    print("Non-zero features:", row.nnz)

    for idx, value in zip(row.indices, row.data):
        print(feature_names[idx], round(float(value), 6))

vector_a = X.getrow(row_a)
vector_b = X.getrow(row_b)

difference = vector_a - vector_b
difference = difference.tocoo()

print("\nORIGINAL VECTOR COMPARISON")
print("Exact match:", (vector_a != vector_b).nnz == 0)
print("Different feature count:", difference.nnz)

for idx, value in zip(difference.col, difference.data):
    print(
        feature_names[idx],
        "difference:",
        round(float(value), 6),
        "temporal:",
        feature_names[idx] in temporal_features
    )

keep_indices = [
    i for i, name in enumerate(feature_names)
    if name not in temporal_features
]

reduced_a = vector_a[:, keep_indices]
reduced_b = vector_b[:, keep_indices]

print("\nAFTER TEMPORAL FEATURE REMOVAL")
print("Exact match:", (reduced_a != reduced_b).nnz == 0)

print("\nDuplicate vector inspection complete.")
