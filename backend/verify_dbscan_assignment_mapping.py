
import pandas as pd

BASE = "dataset/processed/ml/"

metadata = pd.read_csv(BASE + "alert_feature_metadata.csv")
assignments = pd.read_csv(BASE + "dbscan_20k_assignments.csv")

keys = ["OrgId", "AlertId"]

# Check assignment IDs against metadata
mapping = assignments.merge(
    metadata.reset_index(names="matrix_row"),
    on=keys,
    how="left",
    indicator=True,
    validate="one_to_one"
)

print("=" * 55)
print("DBSCAN ASSIGNMENT MAPPING VALIDATION")
print("=" * 55)

print("Total assignments:", len(assignments))
print("Unique assignment keys:",
      assignments[keys].drop_duplicates().shape[0])

print("\nMapping results:")
print(mapping["_merge"].value_counts())

print("\nUnmapped assignments:",
      (mapping["_merge"] != "both").sum())

print("\nMapped matrix row indices:")
print(mapping["matrix_row"].head(10).tolist())

print("\nDuplicate matrix row indices:",
      mapping["matrix_row"].duplicated().sum())

print("\nAll assignments mapped:",
      mapping["matrix_row"].notna().all())

print("\nValidation complete.")

