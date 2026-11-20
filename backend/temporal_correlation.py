
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

FEATURES_PATH = ROOT / "dataset/processed/alert_features.parquet"
ANOMALY_PATH = ROOT / "reports/isolation_forest_assignments.csv"

ASSIGNMENTS_PATH = ROOT / "dataset/processed/temporal_alert_assignments.parquet"
EVENTS_PATH = ROOT / "dataset/processed/temporal_events.parquet"
EVENTS_CSV_PATH = ROOT / "reports/temporal_events.csv"
REPORT_PATH = ROOT / "reports/temporal_correlation_report.json"

MAX_GAP_MINUTES = 5
MAX_EVENT_MINUTES = 60


def main():
    start_time = time.time()

    print("Loading alert features...")
    alerts = pd.read_parquet(FEATURES_PATH)

    alerts["first_seen"] = pd.to_datetime(
        alerts["first_seen"], errors="coerce", utc=True
    )
    alerts["last_seen"] = pd.to_datetime(
        alerts["last_seen"], errors="coerce", utc=True
    )

    alerts = alerts.dropna(
        subset=["OrgId", "AlertId", "first_seen"]
    ).copy()

    print(f"Total alerts: {len(alerts):,}")

    print("Loading anomaly assignments...")
    anomalies = pd.read_csv(ANOMALY_PATH)

    required = {"OrgId", "AlertId"}
    if not required.issubset(anomalies.columns):
        raise ValueError("Anomaly file must contain OrgId and AlertId.")

    if "anomaly_status" not in anomalies.columns:
        if "anomaly" in anomalies.columns:
            anomalies["anomaly_status"] = np.where(
                anomalies["anomaly"].astype(str).str.lower().isin(
                    ["1", "-1", "true", "anomalous"]
                ),
                "Anomalous",
                "Normal"
            )
        else:
            anomalies["anomaly_status"] = "Normal"

    if "anomaly_score" not in anomalies.columns:
        anomalies["anomaly_score"] = np.nan

    for df in [alerts, anomalies]:
        df["OrgId"] = pd.to_numeric(df["OrgId"], errors="coerce")
        df["AlertId"] = pd.to_numeric(df["AlertId"], errors="coerce")

    alerts = alerts.dropna(subset=["OrgId", "AlertId"])
    anomalies = anomalies.dropna(subset=["OrgId", "AlertId"])

    print("Joining anomaly information...")
    alerts = alerts.merge(
        anomalies[[
            "OrgId", "AlertId",
            "anomaly_status", "anomaly_score"
        ]],
        on=["OrgId", "AlertId"],
        how="left",
        validate="one_to_one"
    )

    alerts["anomaly_status"] = alerts["anomaly_status"].fillna("Normal")
    alerts["anomaly_score"] = pd.to_numeric(
        alerts["anomaly_score"], errors="coerce"
    )

    print("Sorting alerts...")
    alerts = alerts.sort_values(
        ["OrgId", "first_seen"],
        kind="mergesort"
    ).reset_index(drop=True)

    print("Generating bounded temporal events...")

    gap_limit = pd.Timedelta(minutes=MAX_GAP_MINUTES)
    duration_limit = pd.Timedelta(minutes=MAX_EVENT_MINUTES)

    event_ids = np.empty(len(alerts), dtype=object)

    for org_id, positions in alerts.groupby(
        "OrgId", sort=False
    ).indices.items():

        positions = np.asarray(positions)
        times = alerts.iloc[positions]["first_seen"].tolist()

        event_number = 0
        event_start = None
        previous_time = None

        for i, position in enumerate(positions):
            current_time = times[i]

            new_event = (
                previous_time is None
                or current_time - previous_time > gap_limit
                or current_time - event_start > duration_limit
            )

            if new_event:
                event_number += 1
                event_start = current_time

            event_ids[position] = f"{org_id}_{event_number}"
            previous_time = current_time

    alerts["event_id"] = event_ids

    alerts["is_anomalous"] = (
        alerts["anomaly_status"].astype(str).str.lower()
        == "anomalous"
    )

    print("Summarizing events...")

    events = (
        alerts.groupby(["OrgId", "event_id"], sort=False)
        .agg(
            alert_count=("AlertId", "size"),
            start_time=("first_seen", "min"),
            end_time=("last_seen", "max"),
            anomalous_alerts=("is_anomalous", "sum"),
            mean_anomaly_score=("anomaly_score", "mean")
        )
        .reset_index()
    )

    events["duration_minutes"] = (
        events["end_time"] - events["start_time"]
    ).dt.total_seconds() / 60

    if "categories" in alerts.columns:
        category_summary = (
            alerts.groupby(["OrgId", "event_id"])["categories"]
            .apply(
                lambda values: sorted({
                    str(item)
                    for value in values.dropna()
                    for item in (
                        value if isinstance(
                            value, (list, tuple, set, np.ndarray)
                        ) else [value]
                    )
                })
            )
            .rename("categories")
            .reset_index()
        )

        events = events.merge(
            category_summary,
            on=["OrgId", "event_id"],
            how="left"
        )

    if "incident_ids" in alerts.columns:
        incident_summary = (
            alerts.groupby(["OrgId", "event_id"])["incident_ids"]
            .apply(
                lambda values: sorted({
                    str(item)
                    for value in values.dropna()
                    for item in (
                        value if isinstance(
                            value, (list, tuple, set, np.ndarray)
                        ) else [value]
                    )
                })
            )
            .rename("incident_ids")
            .reset_index()
        )

        events = events.merge(
            incident_summary,
            on=["OrgId", "event_id"],
            how="left"
        )

    events["event_type"] = "Normal"

    events.loc[
        events["alert_count"] >= 5,
        "event_type"
    ] = "AlertBurst"

    events.loc[
        events["anomalous_alerts"] > 0,
        "event_type"
    ] = "AnomalyLinked"

    events.loc[
        (events["alert_count"] >= 5)
        & (events["anomalous_alerts"] > 0),
        "event_type"
    ] = "HighAttention"

    print("Saving outputs...")

    ASSIGNMENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVENTS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    alerts.to_parquet(ASSIGNMENTS_PATH, index=False)
    events.to_parquet(EVENTS_PATH, index=False)
    events.to_csv(EVENTS_CSV_PATH, index=False)

    report = {
        "total_alerts": int(len(alerts)),
        "total_events": int(len(events)),
        "multi_alert_events": int(
            (events["alert_count"] > 1).sum()
        ),
        "burst_events": int(
            (events["alert_count"] >= 5).sum()
        ),
        "anomaly_linked_events": int(
            (events["anomalous_alerts"] > 0).sum()
        ),
        "high_attention_events": int(
            (events["event_type"] == "HighAttention").sum()
        ),
        "total_anomalous_alerts": int(
            alerts["is_anomalous"].sum()
        ),
        "max_gap_minutes": MAX_GAP_MINUTES,
        "max_event_duration_minutes": MAX_EVENT_MINUTES,
        "largest_event_alert_count": int(
            events["alert_count"].max()
        ),
        "execution_seconds": round(
            time.time() - start_time, 2
        )
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, default=str)

    print("\n========== TEMPORAL CORRELATION SUMMARY ==========")

    for key, value in report.items():
        print(f"{key}: {value}")

    print("\nAll outputs saved successfully.")


if __name__ == "__main__":
    main()
