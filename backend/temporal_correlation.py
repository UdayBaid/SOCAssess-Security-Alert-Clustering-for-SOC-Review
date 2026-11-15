
import json
import time
import pandas as pd

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dataset" / "processed"
ML_DIR = DATA_DIR / "ml"
REPORT_DIR = ROOT / "reports"

REPORT_DIR.mkdir(exist_ok=True)

start = time.time()
WINDOW_MINUTES = 5

print("Loading alert features...")

alerts = pd.read_parquet(
    DATA_DIR / "alert_features.parquet"
)

anomalies = pd.read_csv(
    REPORT_DIR / "isolation_forest_assignments.csv"
)

alerts["OrgId"] = pd.to_numeric(alerts["OrgId"])
alerts["AlertId"] = pd.to_numeric(alerts["AlertId"])

anomalies["OrgId"] = pd.to_numeric(anomalies["OrgId"])
anomalies["AlertId"] = pd.to_numeric(anomalies["AlertId"])

alerts["first_seen"] = pd.to_datetime(
    alerts["first_seen"], errors="coerce"
)

alerts["last_seen"] = pd.to_datetime(
    alerts["last_seen"], errors="coerce"
)

print("Total alerts:", len(alerts))

print("\nAttaching anomaly information...")

alerts = alerts.merge(
    anomalies[
        ["OrgId", "AlertId", "anomaly_status", "anomaly_score"]
    ],
    on=["OrgId", "AlertId"],
    how="left",
    validate="one_to_one"
)

alerts["is_anomalous"] = (
    alerts["anomaly_status"] == "Anomalous"
).astype(int)

print("\nSorting alerts by organization and timestamp...")

alerts = alerts.sort_values(
    ["OrgId", "first_seen"]
).reset_index(drop=True)

print("\nBuilding temporal event groups...")

time_gap = (
    alerts.groupby("OrgId")["first_seen"]
    .diff()
    .dt.total_seconds()
    .div(60)
)

new_event = time_gap.isna() | (time_gap > WINDOW_MINUTES)

session_number = (
    new_event.astype(int)
    .groupby(alerts["OrgId"])
    .cumsum()
)

alerts["temporal_event_id"] = (
    alerts["OrgId"].astype(str)
    + "_"
    + session_number.astype(str)
)

alerts["time_gap_minutes"] = time_gap

print("\nGenerating event summaries...")

events = (
    alerts.groupby("temporal_event_id")
    .agg(
        OrgId=("OrgId", "first"),
        alert_count=("AlertId", "count"),
        first_seen=("first_seen", "min"),
        last_seen=("last_seen", "max"),
        anomalous_alerts=("is_anomalous", "sum"),
        avg_anomaly_score=("anomaly_score", "mean"),
        total_evidence=("evidence_count", "sum"),
        unique_incidents=("incident_ids", "first")
    )
    .reset_index()
)

events["duration_minutes"] = (
    events["last_seen"] - events["first_seen"]
).dt.total_seconds() / 60

events["anomaly_percentage"] = (
    events["anomalous_alerts"]
    / events["alert_count"] * 100
).round(2)

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

print("\nTEMPORAL CORRELATION SUMMARY")

print("Total alerts:", len(alerts))
print("Total temporal events:", len(events))
print(
    "Events containing multiple alerts:",
    int((events["alert_count"] > 1).sum())
)
print(
    "Alert burst events:",
    int((events["alert_count"] >= 5).sum())
)
print(
    "Events linked to anomalies:",
    int((events["anomalous_alerts"] > 0).sum())
)
print(
    "High attention events:",
    int((events["event_type"] == "HighAttention").sum())
)

print("\nTOP 15 LARGEST EVENTS")

print(
    events.sort_values(
        "alert_count", ascending=False
    ).head(15).to_string(index=False)
)

alerts.to_parquet(
    DATA_DIR / "temporal_alert_assignments.parquet",
    index=False
)

events.to_parquet(
    DATA_DIR / "temporal_events.parquet",
    index=False
)

events.to_csv(
    REPORT_DIR / "temporal_events.csv",
    index=False
)

report = {
    "total_alerts": int(len(alerts)),
    "total_temporal_events": int(len(events)),
    "multi_alert_events": int((events["alert_count"] > 1).sum()),
    "alert_burst_events": int((events["alert_count"] >= 5).sum()),
    "anomaly_linked_events": int((events["anomalous_alerts"] > 0).sum()),
    "high_attention_events": int((events["event_type"] == "HighAttention").sum()),
    "window_minutes": WINDOW_MINUTES,
    "execution_seconds": round(time.time() - start, 2)
}

with open(
    REPORT_DIR / "temporal_correlation_report.json", "w"
) as f:
    json.dump(report, f, indent=4, default=str)

print("\nTemporal correlation completed.")
print("Assignments saved.")
print("Event summaries saved.")
print("Report saved.")
print("Execution time:", report["execution_seconds"])
