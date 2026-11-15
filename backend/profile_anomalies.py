
import json
import re
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reports"

anomalies = pd.read_csv(
    REPORT_DIR / "isolation_forest_assignments.csv"
)

features = pd.read_parquet(
    ROOT / "dataset" / "processed" / "alert_features.parquet"
)

# Match alerts using their original identifiers
for df in [anomalies, features]:
    df["OrgId"] = pd.to_numeric(df["OrgId"])
    df["AlertId"] = pd.to_numeric(df["AlertId"])

merged = anomalies.merge(
    features,
    on=["OrgId", "AlertId"],
    how="left",
    validate="one_to_one"
)

anomalous = merged[
    merged["anomaly_status"] == "Anomalous"
].copy()

anomalous = anomalous.sort_values("anomaly_score")

def extract_values(value):
    if pd.isna(value):
        return []

    if isinstance(value, (list, tuple, set)):
        return [str(x).strip() for x in value if str(x).strip()]

    value = str(value).strip()

    if not value:
        return []

    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed]
    except (ValueError, TypeError):
        pass

    return [
        x.strip()
        for x in re.split(r"[;,|]", value)
        if x.strip()
    ]

print("\nANOMALY PROFILE")
print("Total anomalies:", len(anomalous))

numeric_columns = [
    "evidence_count",
    "unique_entity_types",
    "unique_devices",
    "unique_ip_addresses",
    "unique_accounts",
    "unique_file_hashes",
    "unique_urls"
]

print("\nNUMERIC FEATURE SUMMARY")
print(anomalous[numeric_columns].describe().round(2))

for column in ["categories", "mitre_techniques"]:
    counts = (
        anomalous[column]
        .apply(extract_values)
        .explode()
        .dropna()
        .value_counts()
    )

    print(f"\nTOP {column.upper()}")
    print(counts.head(15))

output_columns = [
    "OrgId",
    "AlertId",
    "anomaly_score",
    "evidence_count",
    "categories",
    "mitre_techniques",
    "unique_devices",
    "unique_ip_addresses",
    "unique_accounts",
    "unique_file_hashes",
    "unique_urls"
]

anomalous[output_columns].to_csv(
    REPORT_DIR / "anomaly_profile.csv",
    index=False
)

print("\nTop 20 most anomalous alerts:")
print(anomalous[output_columns].head(20).to_string(index=False))

print("\nAnomaly profile saved.")
