
import json
import pandas as pd
from scipy.sparse import load_npz

BASE = "dataset/processed/ml/"

assignments = pd.read_csv(BASE + "dbscan_20k_assignments.csv")
metadata = pd.read_csv(BASE + "alert_feature_metadata.csv")
X = load_npz(BASE + "alert_features_matrix.npz")

with open(BASE + "feature_names.json", "r", encoding="utf-8") as f:
    feature_names = json.load(f)

if isinstance(feature_names, dict):
    feature_names = feature_names["feature_names"]

full_features = pd.read_parquet(
    "dataset/processed/alert_features.parquet"
)

print("=" * 55)
print("FEATURE ALIGNMENT VALIDATION")
print("=" * 55)

print("Matrix rows:", X.shape[0])
print("Metadata rows:", len(metadata))
print("Feature names:", len(feature_names))
print("Matrix columns:", X.shape[1])

full_features_sorted = full_features.sort_values(
    ["OrgId", "AlertId"]
).reset_index(drop=True)

metadata_sorted = metadata.sort_values(
    ["OrgId", "AlertId"]
).reset_index(drop=True)

metadata_matches = metadata_sorted.equals(
    full_features_sorted[["OrgId", "AlertId"]].reset_index(drop=True)
)

print("\nMetadata key-set matches feature table:", metadata_matches)

print("\nMetadata order matches feature table:", metadata_matches)
print("Feature count matches:", len(feature_names) == X.shape[1])

assignment_matches = assignments[["OrgId", "AlertId"]].merge(
    metadata[["OrgId", "AlertId"]],
    on=["OrgId", "AlertId"],
    how="left",
    indicator=True
)

print(
    "All assigned alerts found in metadata:",
    (assignment_matches["_merge"] == "both").all()
)

print("\nAssignment columns:", list(assignments.columns))
print("Assignment count:", len(assignments))
print(
    "Unique assigned alerts:",
    assignments[["OrgId", "AlertId"]].drop_duplicates().shape[0]
)

print("\nCluster distribution:")
print(assignments["cluster_id"].value_counts().sort_index())

print("\nValidation complete.")
