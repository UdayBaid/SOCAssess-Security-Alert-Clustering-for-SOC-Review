
from pathlib import Path
import json
import time

import duckdb
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, hstack, save_npz
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer, StandardScaler


ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "dataset" / "processed" / "alert_features.parquet"
OUTPUT_DIR = ROOT / "dataset" / "processed" / "ml"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

start = time.time()

print("=" * 60)
print("SOCAssess - ML FEATURE MATRIX GENERATION")
print("=" * 60)


con = duckdb.connect()
con.execute("SET memory_limit='2GB'")
con.execute("SET threads=2")

columns = [
    "OrgId",
    "AlertId",
    "evidence_count",
    "first_seen_hour",
    "first_seen_day_of_week",
    "is_weekend",
    "unique_entity_types",
    "unique_devices",
    "unique_ip_addresses",
    "unique_accounts",
    "unique_file_hashes",
    "unique_urls",
    "categories",
    "alert_titles",
    "mitre_techniques",
]

query = f"""
SELECT {", ".join(columns)}
FROM read_parquet('{INPUT_FILE.as_posix()}')
ORDER BY OrgId, AlertId
"""

df = con.execute(query).df()
con.close()

print("Loaded alerts:", len(df))


metadata = df[["OrgId", "AlertId"]].copy()


count_columns = [
    "evidence_count",
    "unique_entity_types",
    "unique_devices",
    "unique_ip_addresses",
    "unique_accounts",
    "unique_file_hashes",
    "unique_urls",
]

numeric = df[count_columns].astype(np.float32)

numeric = np.log1p(numeric)


title_counts = df["alert_titles"].value_counts()

title_frequency = (
    df["alert_titles"]
    .map(title_counts)
    .astype(np.float32)
)

numeric["alert_title_frequency"] = np.log1p(title_frequency)

scaler = StandardScaler()
numeric_scaled = scaler.fit_transform(numeric)

numeric_matrix = csr_matrix(numeric_scaled)


hour = df["first_seen_hour"].to_numpy(dtype=np.float32)
day = df["first_seen_day_of_week"].to_numpy(dtype=np.float32)

temporal = np.column_stack([
    np.sin(2 * np.pi * hour / 24),
    np.cos(2 * np.pi * hour / 24),
    np.sin(2 * np.pi * day / 7),
    np.cos(2 * np.pi * day / 7),
    df["is_weekend"].to_numpy(dtype=np.float32),
])

temporal_matrix = csr_matrix(temporal)


categories = df["categories"].fillna("UNKNOWN").astype(str)

category_encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=True,
    dtype=np.float32
)

category_matrix = category_encoder.fit_transform(
    categories.to_frame()
)


mitre_lists = (
    df["mitre_techniques"]
    .fillna("")
    .apply(
        lambda value: [
            item.strip()
            for item in value.split(";")
            if item.strip()
        ]
    )
    .tolist()
)

has_mitre = np.array(
    [bool(items) for items in mitre_lists],
    dtype=np.float32
).reshape(-1, 1)

mitre_encoder = MultiLabelBinarizer(sparse_output=True)

mitre_matrix = mitre_encoder.fit_transform(mitre_lists)
mitre_matrix = mitre_matrix.astype(np.float32)

has_mitre_matrix = csr_matrix(has_mitre)


X = hstack(
    [
        numeric_matrix,
        temporal_matrix,
        category_matrix,
        mitre_matrix,
        has_mitre_matrix,
    ],
    format="csr",
    dtype=np.float32
)


numeric_names = list(numeric.columns)

temporal_names = [
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "is_weekend",
]

category_names = [
    f"category_{value}"
    for value in category_encoder.categories_[0]
]

mitre_names = [
    f"mitre_{value}"
    for value in mitre_encoder.classes_
]

feature_names = (
    numeric_names
    + temporal_names
    + category_names
    + mitre_names
    + ["has_mitre"]
)

save_npz(OUTPUT_DIR / "alert_features_matrix.npz", X)

metadata.to_csv(
    OUTPUT_DIR / "alert_feature_metadata.csv",
    index=False
)

with open(OUTPUT_DIR / "feature_names.json", "w") as file:
    json.dump(feature_names, file, indent=2)

with open(OUTPUT_DIR / "mitre_vocabulary.json", "w") as file:
    json.dump(mitre_encoder.classes_.tolist(), file, indent=2)

with open(OUTPUT_DIR / "title_frequency.json", "w") as file:
    json.dump(
        {
            str(key): int(value)
            for key, value in title_counts.items()
        },
        file,
        indent=2
    )

print("\n" + "=" * 60)
print("FEATURE MATRIX SUMMARY")
print("=" * 60)

print("Alerts:", X.shape[0])
print("Features:", X.shape[1])
print("Non-zero values:", X.nnz)
print("Matrix format:", X.format)
print("Matrix data type:", X.dtype)
print("Sparse matrix size (MB):",
      round(
          (X.data.nbytes + X.indices.nbytes + X.indptr.nbytes)
          / (1024 ** 2), 2
      ))

print("MITRE techniques:", len(mitre_encoder.classes_))
print("Categories:", len(category_encoder.categories_[0]))

print("\nValidation:")
print("Rows match metadata:", X.shape[0] == len(metadata))
print("Feature names match:", X.shape[1] == len(feature_names))
print("All values finite:", np.isfinite(X.data).all())

print("\nSaved directory:", OUTPUT_DIR)
print("Time taken:", round(time.time() - start, 2), "seconds")

print("\nFEATURE MATRIX GENERATION COMPLETE")
