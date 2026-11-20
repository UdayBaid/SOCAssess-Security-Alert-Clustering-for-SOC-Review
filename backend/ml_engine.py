
from pathlib import Path
import json
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================
# PATH CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATASET = ROOT / "dataset"
PROCESSED = DATASET / "processed"
ML_DIR = PROCESSED / "ml"
REPORTS = ROOT / "reports"

ALERT_FEATURES = PROCESSED / "alert_features.parquet"

KMEANS_FILE = ML_DIR / "kmeans_k3_assignments.csv"
DBSCAN_FILE = ML_DIR / "dbscan_20k_assignments.csv"
ISOLATION_FILE = REPORTS / "isolation_forest_assignments.csv"

TEMPORAL_ASSIGNMENTS = PROCESSED / "temporal_alert_assignments.parquet"
TEMPORAL_EVENTS = PROCESSED / "temporal_events.parquet"

OUTPUT_PARQUET = PROCESSED / "socassess_ml_results.parquet"
OUTPUT_CSV = REPORTS / "socassess_ml_results.csv"
OUTPUT_REPORT = REPORTS / "socassess_ml_engine_report.json"


# ============================================================
# GENERAL HELPERS
# ============================================================

def print_section(title):
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)


def load_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported file format: {path}")


def find_column(df, candidates, required=True):
    for column in candidates:
        if column in df.columns:
            return column

    if required:
        raise KeyError(
            f"Could not find any of these columns: {candidates}\n"
            f"Available columns: {list(df.columns)}"
        )

    return None


def prepare_keys(df):
    df = df.copy()

    required = ["OrgId", "AlertId"]

    for col in required:
        if col not in df.columns:
            raise KeyError(f"Missing merge key: {col}")

        df[col] = df[col].astype("string").str.strip()

    return df


def safe_numeric(series, default=0):
    return pd.to_numeric(series, errors="coerce").fillna(default)


def safe_bool(series):
    if series.dtype == bool:
        return series.astype(int)

    return (
        series.astype(str)
        .str.lower()
        .isin(["true", "1", "yes", "anomalous", "high", "critical"])
        .astype(int)
    )


def merge_assignment(base, assignment, value_name, candidates):
    assignment = prepare_keys(assignment)

    source_col = find_column(assignment, candidates)

    assignment = assignment[
        ["OrgId", "AlertId", source_col]
    ].copy()

    assignment = assignment.rename(
        columns={source_col: value_name}
    )

    assignment = assignment.drop_duplicates(
        subset=["OrgId", "AlertId"],
        keep="first"
    )

    base = base.merge(
        assignment,
        on=["OrgId", "AlertId"],
        how="left",
        validate="one_to_one"
    )

    return base


# ============================================================
# KMEANS INTEGRATION
# ============================================================

def integrate_kmeans(alerts):
    print("Integrating KMeans assignments...")

    assignments = load_file(KMEANS_FILE)

    alerts = merge_assignment(
        alerts,
        assignments,
        "kmeans_cluster",
        ["cluster_id", "cluster", "kmeans_cluster", "label", "labels"]
    )

    alerts["kmeans_cluster"] = pd.to_numeric(
        alerts["kmeans_cluster"],
        errors="coerce"
    )

    print(
        "KMeans mapped alerts:",
        int(alerts["kmeans_cluster"].notna().sum())
    )

    return alerts


# ============================================================
# DBSCAN INTEGRATION
# ============================================================

def integrate_dbscan(alerts):
    print("Integrating DBSCAN assignments...")

    assignments = load_file(DBSCAN_FILE)

    alerts = merge_assignment(
        alerts,
        assignments,
        "dbscan_cluster",
        ["cluster_id", "cluster", "dbscan_cluster", "label", "labels"]
    )

    alerts["dbscan_cluster"] = pd.to_numeric(
        alerts["dbscan_cluster"],
        errors="coerce"
    )

    # DBSCAN label -1 represents noise.
    alerts["signal_dbscan_noise"] = (
        alerts["dbscan_cluster"].eq(-1).astype(int)
    )

    print(
        "DBSCAN mapped alerts:",
        int(alerts["dbscan_cluster"].notna().sum())
    )

    print(
        "DBSCAN noise alerts:",
        int(alerts["signal_dbscan_noise"].sum())
    )

    return alerts


# ============================================================
# ISOLATION FOREST INTEGRATION
# ============================================================

def integrate_isolation_forest(alerts):
    print("Integrating Isolation Forest assignments...")

    assignments = load_file(ISOLATION_FILE)

    assignments = prepare_keys(assignments)

    status_col = find_column(
        assignments,
        [
            "anomaly_status",
            "status",
            "prediction_label",
            "anomaly_label",
            "is_anomaly"
        ],
        required=False
    )

    score_col = find_column(
        assignments,
        [
            "anomaly_score",
            "decision_score",
            "score",
            "anomaly_score_samples"
        ],
        required=False
    )

    keep_cols = ["OrgId", "AlertId"]

    if status_col:
        keep_cols.append(status_col)

    if score_col and score_col not in keep_cols:
        keep_cols.append(score_col)

    assignments = assignments[keep_cols].copy()

    assignments = assignments.drop_duplicates(
        subset=["OrgId", "AlertId"],
        keep="first"
    )

    rename = {}

    if status_col:
        rename[status_col] = "anomaly_status"

    if score_col:
        rename[score_col] = "anomaly_score"

    assignments = assignments.rename(columns=rename)

    alerts = alerts.merge(
        assignments,
        on=["OrgId", "AlertId"],
        how="left",
        validate="one_to_one"
    )

    if "anomaly_status" in alerts.columns:
        status = alerts["anomaly_status"]

        if pd.api.types.is_numeric_dtype(status):
            alerts["signal_anomaly"] = (
                pd.to_numeric(status, errors="coerce")
                .eq(-1)
                .astype(int)
            )
        else:
            alerts["signal_anomaly"] = (
                status.astype(str)
                .str.lower()
                .isin(["anomalous", "anomaly", "true", "1", "-1"])
                .astype(int)
            )
    else:
        alerts["signal_anomaly"] = 0

    if "anomaly_score" in alerts.columns:
        alerts["anomaly_score"] = safe_numeric(
            alerts["anomaly_score"],
            default=np.nan
        )

    print(
        "Anomalous alerts:",
        int(alerts["signal_anomaly"].sum())
    )

    return alerts


# ============================================================
# TEMPORAL CORRELATION INTEGRATION
# ============================================================


def integrate_temporal(alerts):
    print("Integrating temporal correlation...")

    assignments = load_file(TEMPORAL_ASSIGNMENTS)
    events = load_file(TEMPORAL_EVENTS)

    assignments = prepare_keys(assignments)

    event_id_col = find_column(
        assignments,
        ["event_id", "temporal_event_id", "event"]
    )

    assignments = assignments[
        ["OrgId", "AlertId", event_id_col]
    ].copy()

    assignments = assignments.rename(
        columns={event_id_col: "event_id"}
    )

    assignments = assignments.drop_duplicates(
        subset=["OrgId", "AlertId"],
        keep="first"
    )

    alerts = alerts.merge(
        assignments,
        on=["OrgId", "AlertId"],
        how="left",
        validate="one_to_one"
    )

    events = events.copy()
    events["event_id"] = events["event_id"].astype("string")
    alerts["event_id"] = alerts["event_id"].astype("string")

    # Actual temporal event fields
    events["event_type"] = (
        events["event_type"].astype(str)
    )

    event_summary = events[
        [
            "event_id",
            "alert_count",
            "anomalous_alerts",
            "mean_anomaly_score",
            "event_type"
        ]
    ].drop_duplicates(
        subset=["event_id"],
        keep="first"
    )

    event_summary = event_summary.rename(
        columns={
            "alert_count": "event_alert_count",
            "anomalous_alerts": "event_anomalous_alerts"
        }
    )

    alerts = alerts.merge(
        event_summary,
        on="event_id",
        how="left",
        validate="many_to_one"
    )

    alerts["event_alert_count"] = safe_numeric(
        alerts["event_alert_count"]
    )

    alerts["event_anomalous_alerts"] = safe_numeric(
        alerts["event_anomalous_alerts"]
    )

    # Correct event type signals
    alerts["signal_high_attention_event"] = (
        alerts["event_type"].eq("HighAttention").astype(int)
    )

    alerts["signal_alert_burst"] = (
        alerts["event_type"].eq("AlertBurst").astype(int)
    )

    alerts["signal_anomaly_linked_event"] = (
        alerts["event_type"].eq("AnomalyLinked").astype(int)
    )

    print(
        "Alerts assigned to temporal events:",
        int(alerts["event_id"].notna().sum())
    )

    print(
        "High-attention event alerts:",
        int(alerts["signal_high_attention_event"].sum())
    )

    print(
        "Alert burst event alerts:",
        int(alerts["signal_alert_burst"].sum())
    )

    print(
        "Anomaly-linked event alerts:",
        int(alerts["signal_anomaly_linked_event"].sum())
    )

    return alerts



# ============================================================
# PRIORITY SCORING
# ============================================================

def calculate_priority(alerts):
    print("Calculating SOC priority scores...")

    # Evidence volume: use an available evidence/count feature.
    evidence_col = find_column(
        alerts,
        [
            "evidence_count",
            "EvidenceCount",
            "total_evidence",
            "record_count",
            "alert_count"
        ],
        required=False
    )

    if evidence_col:
        alerts["evidence_count_for_score"] = safe_numeric(
            alerts[evidence_col]
        )
    else:
        alerts["evidence_count_for_score"] = 0

    threshold = alerts["evidence_count_for_score"].quantile(0.95)

    alerts["signal_high_evidence"] = (
        alerts["evidence_count_for_score"].ge(threshold)
        & alerts["evidence_count_for_score"].gt(0)
    ).astype(int)

    # Transparent heuristic score, not an attack probability.
    alerts["priority_score"] = (
        alerts["signal_anomaly"] * 40
        + alerts["signal_high_attention_event"] * 25
        + alerts["signal_alert_burst"] * 15
        + alerts["signal_dbscan_noise"] * 10
        + alerts["signal_high_evidence"] * 10
    ).clip(0, 100)

    alerts["priority_level"] = pd.cut(
        alerts["priority_score"],
        bins=[-1, 19, 39, 69, 100],
        labels=["Low", "Medium", "High", "Critical"]
    ).astype(str)

    # Generate explanations without slow row-wise apply.
    reasons = pd.Series(
        "",
        index=alerts.index,
        dtype="object"
    )

    def add_reason(mask, reason):
        nonlocal reasons
        reasons = reasons.where(
            ~mask,
            reasons + reason + "; "
        )

    add_reason(
        alerts["signal_anomaly"].eq(1),
        "Isolation Forest anomaly"
    )

    add_reason(
        alerts["signal_high_attention_event"].eq(1),
        "High-attention temporal event"
    )

    add_reason(
        alerts["signal_high_attention_event"].eq(0)
        & alerts["signal_alert_burst"].eq(1),
        "Temporal alert burst"
    )

    add_reason(
        alerts["signal_dbscan_noise"].eq(1),
        "DBSCAN noise point in evaluated sample"
    )

    add_reason(
        alerts["signal_high_evidence"].eq(1),
        "High evidence volume"
    )

    alerts["priority_reasons"] = reasons.str.rstrip("; ")

    alerts.loc[
        alerts["priority_reasons"].eq(""),
        "priority_reasons"
    ] = "No elevated signals detected"

    return alerts


# ============================================================
# REPORT GENERATION
# ============================================================

def generate_report(alerts, execution_seconds):
    print("Generating unified ML report...")

    priority_counts = (
        alerts["priority_level"]
        .value_counts()
        .to_dict()
    )

    report = {
        "project": "SOCAssess",
        "module": "Unified ML Engine",
        "total_alerts": int(len(alerts)),
        "kmeans_mapped_alerts": int(
            alerts["kmeans_cluster"].notna().sum()
        ),
        "dbscan_mapped_alerts": int(
            alerts["dbscan_cluster"].notna().sum()
        ),
        "isolation_forest_anomalies": int(
            alerts["signal_anomaly"].sum()
        ),
        "temporal_event_alerts": int(
            alerts["event_id"].notna().sum()
        ),
        "high_attention_alerts": int(
            alerts["signal_high_attention_event"].sum()
        ),
        "dbscan_noise_alerts": int(
            alerts["signal_dbscan_noise"].sum()
        ),
        "priority_distribution": {
            str(key): int(value)
            for key, value in priority_counts.items()
        },
        "priority_score_method": {
            "isolation_forest_anomaly": 40,
            "high_attention_temporal_event": 25,
            "temporal_alert_burst": 15,
            "dbscan_noise": 10,
            "high_evidence_volume": 10
        },
        "important_limitations": [
            "Priority score is a heuristic decision-support score, not attack probability.",
            "Isolation Forest anomalies are not confirmed attacks.",
            "DBSCAN assignments cover only the evaluated 20,000-alert sample.",
            "KMeans and DBSCAN clusters are exploratory groupings, not ground-truth labels.",
            "High-priority alerts require human SOC analyst investigation."
        ],
        "execution_seconds": round(execution_seconds, 2)
    }

    with open(
        OUTPUT_REPORT,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=4)

    return report


# ============================================================
# MAIN ENGINE
# ============================================================

def main():
    start_time = time.time()

    print_section("SOCAssess UNIFIED ML ENGINE")

    # Create output directories.
    PROCESSED.mkdir(parents=True, exist_ok=True)
    ML_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    # Step 1: Load base alert features.
    print("Loading alert features...")

    alerts = load_file(ALERT_FEATURES)
    alerts = prepare_keys(alerts)

    alerts = alerts.drop_duplicates(
        subset=["OrgId", "AlertId"],
        keep="first"
    ).reset_index(drop=True)

    print("Total alerts:", f"{len(alerts):,}")

    # Step 2: Integrate ML models.
    alerts = integrate_kmeans(alerts)

    alerts = integrate_dbscan(alerts)

    alerts = integrate_isolation_forest(alerts)

    # Step 3: Integrate temporal correlation.
    alerts = integrate_temporal(alerts)

    # Step 4: Calculate priority.
    alerts = calculate_priority(alerts)

    # Step 5: Sort results.
    alerts = alerts.sort_values(
        by=["priority_score", "evidence_count_for_score"],
        ascending=[False, False]
    ).reset_index(drop=True)

    # Step 6: Save complete results.
    print_section("SAVING OUTPUTS")

    alerts.to_parquet(
        OUTPUT_PARQUET,
        index=False
    )

    # CSV is limited to the top 50,000 records for manageable size.
    alerts.head(50000).to_csv(
        OUTPUT_CSV,
        index=False
    )

    execution_seconds = time.time() - start_time

    report = generate_report(
        alerts,
        execution_seconds
    )

    print_section("UNIFIED ML ENGINE SUMMARY")

    print("Total alerts:", f"{report['total_alerts']:,}")
    print("KMeans mapped:", f"{report['kmeans_mapped_alerts']:,}")
    print("DBSCAN mapped:", f"{report['dbscan_mapped_alerts']:,}")
    print(
        "Isolation Forest anomalies:",
        f"{report['isolation_forest_anomalies']:,}"
    )
    print(
        "Temporal event alerts:",
        f"{report['temporal_event_alerts']:,}"
    )

    print("\nPRIORITY DISTRIBUTION")

    for level, count in report["priority_distribution"].items():
        print(f"{level}: {count:,}")

    print("\nExecution time:", round(execution_seconds, 2), "seconds")

    print("\nOUTPUT FILES")
    print(OUTPUT_PARQUET)
    print(OUTPUT_CSV)
    print(OUTPUT_REPORT)

    print_section("ENGINE EXECUTION COMPLETED")


if __name__ == "__main__":
    main()
