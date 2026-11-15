
import pandas as pd

metadata = pd.read_csv(
    "dataset/processed/ml/alert_feature_metadata.csv"
)

features = pd.read_parquet(
    "dataset/processed/alert_features.parquet"
)

keys = ["OrgId", "AlertId"]

print("METADATA FIRST 5 ROWS")
print(metadata[keys].head().to_string(index=False))

print("\nFEATURE TABLE FIRST 5 ROWS")
print(features[keys].head().to_string(index=False))

print("\nROW ORDER MATCH:")
print(metadata[keys].equals(features[keys]))

print("\nDUPLICATE KEY CHECK:")
print("Metadata duplicates:", metadata.duplicated(keys).sum())
print("Feature duplicates:", features.duplicated(keys).sum())

metadata_sorted = metadata[keys].sort_values(keys).reset_index(drop=True)
features_sorted = features[keys].sort_values(keys).reset_index(drop=True)

print("\nSORTED KEY MATCH:")
print(metadata_sorted.equals(features_sorted))

print("\nFIRST 10 POSITION MISMATCHES:")
comparison = metadata[keys].reset_index(drop=True).compare(
    features[keys].reset_index(drop=True)
)

print(comparison.head(10).to_string())
