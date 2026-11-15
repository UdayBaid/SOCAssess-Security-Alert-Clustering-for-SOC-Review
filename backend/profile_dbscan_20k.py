
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURES_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "alert_features.parquet"
)

ASSIGNMENTS_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "ml"
    / "dbscan_20k_assignments.csv"
)

REPORT_DIR = PROJECT_ROOT / "reports"


def split_values(value):
    if value is None:
        return []

    if isinstance(value, float) and np.isnan(value):
        return []

    if isinstance(value, (list, tuple, set, np.ndarray)):
        values = value
    else:
        values = str(value).split(";")

    return [
        str(item).strip()
        for item in values
        if str(item).strip()
    ]


def main():

    print("Loading alert features...")
    features = pd.read_parquet(FEATURES_FILE)

    print("Loading DBSCAN assignments...")
    assignments = pd.read_csv(ASSIGNMENTS_FILE)

    required_keys = {"OrgId", "AlertId", "cluster_id"}

    if not required_keys.issubset(assignments.columns):
        raise ValueError(
            f"Assignment file must contain {required_keys}"
        )

    for df in [features, assignments]:
        df["OrgId"] = pd.to_numeric(
            df["OrgId"], errors="raise"
        ).astype("int64")

        df["AlertId"] = pd.to_numeric(
            df["AlertId"], errors="raise"
        ).astype("int64")

    if assignments.duplicated(["OrgId", "AlertId"]).any():
        raise ValueError("Duplicate alert keys in assignments.")

    data = assignments.merge(
        features,
        on=["OrgId", "AlertId"],
        how="left",
        validate="one_to_one",
        indicator=True
    )

    unmatched = int((data["_merge"] != "both").sum())

    if unmatched:
        raise ValueError(
            f"{unmatched} assignments could not be matched."
        )

    data.drop(columns=["_merge"], inplace=True)

    print("Matched alerts:", len(data))
    print("Unique alerts:", data[["OrgId", "AlertId"]].drop_duplicates().shape[0])

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 1. CLUSTER SUMMARY
    # --------------------------------------------------

    clustered = data[data["cluster_id"] != -1].copy()

    summary = (
        clustered.groupby("cluster_id")
        .agg(
            alert_count=("AlertId", "count"),
            mean_evidence_count=("evidence_count", "mean"),
            median_evidence_count=("evidence_count", "median"),
            mean_unique_devices=("unique_devices", "mean"),
            mean_unique_ip_addresses=("unique_ip_addresses", "mean"),
            mean_unique_accounts=("unique_accounts", "mean"),
            mean_unique_file_hashes=("unique_file_hashes", "mean"),
            mean_unique_urls=("unique_urls", "mean")
        )
        .reset_index()
    )

    summary["percentage_of_sample"] = (
        summary["alert_count"] / len(data) * 100
    )

    summary = summary.sort_values(
        "alert_count", ascending=False
    )

    summary.to_csv(
        REPORT_DIR / "dbscan_20k_cluster_summary.csv",
        index=False
    )

    # --------------------------------------------------
    # 2. CATEGORY PROFILE
    # --------------------------------------------------

    category_rows = []

    for _, row in clustered[
        ["cluster_id", "categories"]
    ].iterrows():

        for category in split_values(row["categories"]):
            category_rows.append({
                "cluster_id": int(row["cluster_id"]),
                "category": category
            })

    category_df = pd.DataFrame(category_rows)

    if not category_df.empty:
        category_summary = (
            category_df.groupby(
                ["cluster_id", "category"]
            )
            .size()
            .reset_index(name="alert_occurrences")
            .sort_values(
                ["cluster_id", "alert_occurrences"],
                ascending=[True, False]
            )
        )

        category_summary.to_csv(
            REPORT_DIR / "dbscan_20k_cluster_categories.csv",
            index=False
        )

    # --------------------------------------------------
    # 3. MITRE PROFILE
    # --------------------------------------------------

    mitre_rows = []

    for _, row in clustered[
        ["cluster_id", "mitre_techniques"]
    ].iterrows():

        for technique in split_values(row["mitre_techniques"]):
            mitre_rows.append({
                "cluster_id": int(row["cluster_id"]),
                "mitre_technique": technique
            })

    mitre_df = pd.DataFrame(mitre_rows)

    if not mitre_df.empty:
        mitre_summary = (
            mitre_df.groupby(
                ["cluster_id", "mitre_technique"]
            )
            .size()
            .reset_index(name="alert_occurrences")
            .sort_values(
                ["cluster_id", "alert_occurrences"],
                ascending=[True, False]
            )
        )

        mitre_summary.to_csv(
            REPORT_DIR / "dbscan_20k_cluster_mitre.csv",
            index=False
        )

    # --------------------------------------------------
    # 4. NOISE SUMMARY
    # --------------------------------------------------

    noise = data[data["cluster_id"] == -1]

    noise_summary = {
        "total_alerts": int(len(data)),
        "clustered_alerts": int(len(clustered)),
        "noise_alerts": int(len(noise)),
        "noise_percentage": round(
            len(noise) / len(data) * 100, 2
        )
    }

    print("\nNOISE SUMMARY")
    print(noise_summary)

    # --------------------------------------------------
    # 5. DISPLAY TOP CLUSTERS
    # --------------------------------------------------

    print("\nTOP 15 CLUSTERS")
    print(
        summary.head(15).to_string(index=False)
    )

    if not category_df.empty:
        print("\nTOP CATEGORY OCCURRENCES")
        print(
            category_summary.groupby("cluster_id")
            .head(3)
            .to_string(index=False)
        )

    if not mitre_df.empty:
        print("\nTOP MITRE TECHNIQUES")
        print(
            mitre_summary.groupby("cluster_id")
            .head(3)
            .to_string(index=False)
        )

    print("\nReports saved:")
    print("reports/dbscan_20k_cluster_summary.csv")
    print("reports/dbscan_20k_cluster_categories.csv")
    print("reports/dbscan_20k_cluster_mitre.csv")

    print("\n20K cluster profiling complete.")


if __name__ == "__main__":
    main()
