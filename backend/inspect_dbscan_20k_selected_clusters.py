
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

FEATURES_FILE = (
    ROOT / "dataset" / "processed"
    / "alert_features.parquet"
)

ASSIGNMENTS_FILE = (
    ROOT / "dataset" / "processed" / "ml"
    / "dbscan_20k_assignments.csv"
)

REPORT_DIR = ROOT / "reports"

SELECTED_CLUSTERS = [0, 6, 16, 27, 50]


def split_values(value):
    if value is None:
        return []

    if isinstance(value, float) and np.isnan(value):
        return []

    return [
        x.strip()
        for x in str(value).split(";")
        if x.strip()
    ]


def main():

    print("Loading data...")

    features = pd.read_parquet(FEATURES_FILE)
    assignments = pd.read_csv(ASSIGNMENTS_FILE)

    for df in [features, assignments]:
        df["OrgId"] = pd.to_numeric(
            df["OrgId"], errors="raise"
        ).astype("int64")

        df["AlertId"] = pd.to_numeric(
            df["AlertId"], errors="raise"
        ).astype("int64")

    data = assignments.merge(
        features,
        on=["OrgId", "AlertId"],
        how="left",
        validate="one_to_one",
        indicator=True
    )

    if (data["_merge"] != "both").any():
        raise ValueError("Some alerts could not be matched.")

    data = data.drop(columns=["_merge"])

    selected = data[
        data["cluster_id"].isin(SELECTED_CLUSTERS)
    ].copy()

    print("Selected alerts:", len(selected))

    # ------------------------------------------------
    # CATEGORY AND MITRE CROSS-PROFILE
    # ------------------------------------------------

    cross_rows = []

    for _, row in selected.iterrows():

        categories = split_values(row["categories"])
        techniques = split_values(row["mitre_techniques"])

        for category in categories:
            for technique in techniques:
                cross_rows.append({
                    "cluster_id": int(row["cluster_id"]),
                    "category": category,
                    "mitre_technique": technique
                })

    cross_df = pd.DataFrame(cross_rows)

    if not cross_df.empty:

        cross_summary = (
            cross_df.groupby(
                ["cluster_id", "category", "mitre_technique"]
            )
            .size()
            .reset_index(name="occurrences")
            .sort_values(
                ["cluster_id", "occurrences"],
                ascending=[True, False]
            )
        )

        cross_summary.to_csv(
            REPORT_DIR / "dbscan_selected_category_mitre.csv",
            index=False
        )

        print("\nCATEGORY-MITRE COMBINATIONS")
        print(cross_summary.groupby("cluster_id").head(15).to_string(index=False))

    # ------------------------------------------------
    # ALERT TITLE PROFILE
    # ------------------------------------------------

    title_rows = []

    for _, row in selected.iterrows():

        for title in split_values(row["alert_titles"]):
            title_rows.append({
                "cluster_id": int(row["cluster_id"]),
                "alert_title": title
            })

    title_df = pd.DataFrame(title_rows)

    if not title_df.empty:

        title_summary = (
            title_df.groupby(
                ["cluster_id", "alert_title"]
            )
            .size()
            .reset_index(name="occurrences")
            .sort_values(
                ["cluster_id", "occurrences"],
                ascending=[True, False]
            )
        )

        title_summary.to_csv(
            REPORT_DIR / "dbscan_selected_alert_titles.csv",
            index=False
        )

        print("\nTOP ALERT TITLE VALUES")
        print(title_summary.groupby("cluster_id").head(10).to_string(index=False))

    # ------------------------------------------------
    # CLUSTER COMPARISON
    # ------------------------------------------------

    numeric_columns = [
        "evidence_count",
        "unique_entity_types",
        "unique_devices",
        "unique_ip_addresses",
        "unique_accounts",
        "unique_file_hashes",
        "unique_urls"
    ]

    comparison = (
        selected.groupby("cluster_id")[numeric_columns]
        .agg(["mean", "median"])
    )

    comparison.to_csv(
        REPORT_DIR / "dbscan_selected_cluster_comparison.csv"
    )

    print("\nSELECTED CLUSTER NUMERIC COMPARISON")
    print(comparison.to_string())

    print("\nSelected cluster inspection complete.")
    print("Reports saved in:", REPORT_DIR)


if __name__ == "__main__":
    main()
