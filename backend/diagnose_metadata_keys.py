
import pandas as pd

BASE = "dataset/processed/ml/"

metadata = pd.read_csv(BASE + "alert_feature_metadata.csv")
features = pd.read_parquet(
    "dataset/processed/alert_features.parquet"
)

keys = ["OrgId", "AlertId"]

print("ORIGINAL DATA TYPES")
print("Metadata:\n", metadata[keys].dtypes)
print("Features:\n", features[keys].dtypes)

# Convert IDs to numeric values
for col in keys:
    metadata[col] = pd.to_numeric(
        metadata[col], errors="coerce"
    )
    features[col] = pd.to_numeric(
        features[col], errors="coerce"
    )

print("\nINVALID ID VALUES")
print("Metadata:", metadata[keys].isna().sum().to_dict())
print("Features:", features[keys].isna().sum().to_dict())

# Remove invalid keys only for this diagnostic
metadata_keys = metadata[keys].dropna()
feature_keys = features[keys].dropna()

comparison = metadata_keys.merge(
    feature_keys,
    on=keys,
    how="outer",
    indicator=True,
    validate="one_to_one"
)

print("\n" + "=" * 55)
print("ALERT KEY SET VALIDATION")
print("=" * 55)

print("Metadata rows:", len(metadata_keys))
print("Feature rows:", len(feature_keys))

print("\nKey comparison:")
print(comparison["_merge"].value_counts())

metadata_only = comparison[
    comparison["_merge"] == "left_only"
]

features_only = comparison[
    comparison["_merge"] == "right_only"
]

print("\nMetadata-only alerts:", len(metadata_only))
print(metadata_only[keys].head(10).to_string(index=False))

print("\nFeature-only alerts:", len(features_only))
print(features_only[keys].head(10).to_string(index=False))

print("\nExact key-set match:",
      len(metadata_only) == 0 and len(features_only) == 0)

print("\nValidation complete.")
