from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "dataset" / "processed"

assignments = pd.read_csv(
    PROCESSED / "ml" / "kmeans_k3_assignments.csv"
)

features = pd.read_parquet(
    PROCESSED / "alert_features.parquet"
)

print("=" * 60)
print("SOCAssess - CLUSTER DATA INSPECTION")
print("=" * 60)

print("\nAssignments shape:", assignments.shape)
print("Feature data shape:", features.shape)

print("\nFeature columns:")
print(features.columns.tolist())

print("\nSample feature records:")
print(features.head(3).to_string(index=False))

print("\nSample cluster assignments:")
print(assignments.head(5).to_string(index=False))

print("\nAssignment cluster distribution:")
print(assignments["cluster_id"].value_counts().sort_index())

print("\nDuplicate assignment keys:",
      assignments.duplicated(["OrgId", "AlertId"]).sum())

print("\nDuplicate feature keys:",
      features.duplicated(["OrgId", "AlertId"]).sum())

print("\nFeature columns available for profiling:")
for column in [
    "categories",
    "evidence_count",
    "unique_devices",
    "unique_ip_addresses",
    "unique_accounts",
    "mitre_techniques",
    "incident_grades"
]:
    print(f"{column}: {column in features.columns}")