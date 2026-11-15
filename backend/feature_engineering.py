
from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "dataset" / "processed" / "alerts_aggregated.parquet"
OUTPUT_DIR = ROOT / "dataset" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "alert_features.parquet"
OUTLIER_FILE = OUTPUT_DIR / "high_evidence_alerts.csv"
TEMP_DIR = ROOT / "temp"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

def safe_path(path):
    return path.resolve().as_posix().replace("'", "''")

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Aggregated data not found: {INPUT_FILE}")

con = duckdb.connect()

con.execute("SET memory_limit='3GB'")
con.execute("SET threads=2")
con.execute("SET preserve_insertion_order=false")
con.execute(f"SET temp_directory='{safe_path(TEMP_DIR)}'")

source = safe_path(INPUT_FILE)
output = safe_path(OUTPUT_FILE)
outliers = safe_path(OUTLIER_FILE)

print("Loading aggregated alerts...")

con.execute(f"""
    CREATE VIEW alerts AS
    SELECT * FROM read_parquet('{source}')
""")


con.execute(f"""
    COPY (
        SELECT
            OrgId,
            AlertId,
            evidence_count,
            categories,
            unique_detectors,
            unique_devices,
            unique_ip_addresses,
            unique_accounts,
            first_seen,
            last_seen
        FROM alerts
        ORDER BY evidence_count DESC
        LIMIT 20
    )
    TO '{outliers}'
    (HEADER, DELIMITER ',')
""")

print("Creating alert-level features...")

con.execute(f"""
    COPY (
        SELECT
            -- Reference identifiers (not ML features)
            OrgId,
            AlertId,
            incident_ids,

            -- Evidence volume
            evidence_count,

            -- Time features
            first_seen,
            last_seen,

            COALESCE(
                EPOCH(last_seen) - EPOCH(first_seen),
                0
            ) AS alert_duration_seconds,

            EXTRACT(HOUR FROM first_seen)
                AS first_seen_hour,

            DAYOFWEEK(first_seen)
                AS first_seen_day_of_week,

            CASE
                WHEN DAYOFWEEK(first_seen) IN (0, 6)
                THEN 1 ELSE 0
            END AS is_weekend,

            -- Entity diversity
            unique_detectors,
            unique_entity_types,
            unique_devices,
            unique_ip_addresses,
            unique_accounts,
            unique_file_hashes,
            unique_urls,

            -- Categorical and text information
            categories,
            alert_titles,
            mitre_techniques,

            -- Evaluation metadata; never use as an ML feature
            incident_grades

        FROM alerts
    )
    TO '{output}'
    (FORMAT PARQUET, COMPRESSION ZSTD)
""")

result = con.execute(f"""
    SELECT
        COUNT(*) AS total_alerts,
        COUNT(*) FILTER (
            WHERE evidence_count IS NULL
        ) AS missing_evidence_counts,
        COUNT(*) FILTER (
            WHERE alert_duration_seconds < 0
        ) AS negative_durations,
        COUNT(*) FILTER (
            WHERE first_seen_hour IS NULL
        ) AS missing_hours,
        MIN(evidence_count),
        MAX(evidence_count)
    FROM read_parquet('{output}')
""").fetchone()

print("\n" + "=" * 55)
print("FEATURE ENGINEERING SUMMARY")
print("=" * 55)

print(f"Total alert records:       {result[0]:,}")
print(f"Missing evidence counts:   {result[1]:,}")
print(f"Negative durations:        {result[2]:,}")
print(f"Missing time values:       {result[3]:,}")
print(f"Minimum evidence count:    {result[4]}")
print(f"Maximum evidence count:    {result[5]}")

print(f"\nFeature dataset: {OUTPUT_FILE}")
print(f"High-evidence alerts: {OUTLIER_FILE}")

con.close()

print("\nFeature engineering completed.")
