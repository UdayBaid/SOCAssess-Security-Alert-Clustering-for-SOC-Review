
from pathlib import Path
import json

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "dataset" / "processed"
REPORTS = ROOT / "reports"

assignments = pd.read_csv(
    PROCESSED / "ml" / "kmeans_k3_assignments.csv"
)

features = pd.read_parquet(
    PROCESSED / "alert_features.parquet"
)

KEYS = ["OrgId", "AlertId"]

feature_columns = KEYS + [
    "evidence_count",
    "unique_devices",
    "unique_ip_addresses",
    "unique_accounts",
    "unique_file_hashes",
    "unique_urls",
    "categories",
    "mitre_techniques",
    "incident_grades"
]

print("=" * 60)
print("SOCAssess - K-MEANS CLUSTER PROFILING")
print("=" * 60)
for df in [assignments, features]:
    df["OrgId"] = pd.to_numeric(
        df["OrgId"], errors="raise"
    ).astype("int64")

    df["AlertId"] = pd.to_numeric(
        df["AlertId"], errors="raise"
    ).astype("int64")
    
data = assignments.merge(
    features[feature_columns],
    on=KEYS,
    how="inner",
    validate="one_to_one"
)

if len(data) != len(assignments):
    raise ValueError("Some cluster assignments did not match alert features.")

print("\nMatched alerts:", len(data))
print("Unique alert keys:", data[KEYS].drop_duplicates().shape[0])

REPORTS.mkdir(parents=True, exist_ok=True)

numeric_columns = [
    "evidence_count",
    "unique_devices",
    "unique_ip_addresses",
    "unique_accounts",
    "unique_file_hashes",
    "unique_urls"
]

summary = data.groupby("cluster_id").agg(
    alert_count=("AlertId", "count"),
    **{
        f"mean_{column}": (column, "mean")
        for column in numeric_columns
    },
    **{
        f"median_{column}": (column, "median")
        for column in numeric_columns
    }
).reset_index()

summary["percentage"] = (
    summary["alert_count"] / len(data) * 100
).round(2)

summary.to_csv(
    REPORTS / "cluster_summary.csv",
    index=False
)

def get_distribution(frame, column, separator):
    records = frame[["cluster_id", column]].dropna().copy()

    records[column] = records[column].astype(str).str.split(
        separator, regex=False
    )

    records = records.explode(column)
    records[column] = records[column].str.strip()
    records = records[records[column] != ""]

    return (
        records.groupby(["cluster_id", column])
        .size()
        .reset_index(name="count")
        .sort_values(
            ["cluster_id", "count"],
            ascending=[True, False]
        )
    )

categories = get_distribution(data, "categories", "|")
categories.to_csv(
    REPORTS / "cluster_categories.csv",
    index=False
)

mitre = get_distribution(data, "mitre_techniques", ";")
mitre.to_csv(
    REPORTS / "cluster_mitre_techniques.csv",
    index=False
)

grades = get_distribution(data, "incident_grades", "|")
grades.to_csv(
    REPORTS / "cluster_incident_grades.csv",
    index=False
)

print("\nCLUSTER SUMMARY")
print(summary.round(2).to_string(index=False))

for cluster_id in sorted(data["cluster_id"].unique()):
    print(f"\n{'=' * 60}")
    print(f"CLUSTER {cluster_id}")
    print("=" * 60)

    print("\nTop categories:")
    print(
        categories[categories["cluster_id"] == cluster_id]
        .head(10)
        .to_string(index=False)
    )

    print("\nTop MITRE techniques:")
    print(
        mitre[mitre["cluster_id"] == cluster_id]
        .head(10)
        .to_string(index=False)
    )

    print("\nIncidentGrade distribution (external reference):")
    print(
        grades[grades["cluster_id"] == cluster_id]
        .to_string(index=False)
    )

print("\nReports saved:")
print(REPORTS / "cluster_summary.csv")
print(REPORTS / "cluster_categories.csv")
print(REPORTS / "cluster_mitre_techniques.csv")
print(REPORTS / "cluster_incident_grades.csv")

print("\nCLUSTER PROFILING COMPLETE")
