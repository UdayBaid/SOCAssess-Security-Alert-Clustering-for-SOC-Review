
from pathlib import Path
import duckdb
import numpy as np
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = PROJECT_ROOT / "dataset" / "GUIDE_Test.csv"
OUTPUT_DIR = PROJECT_ROOT / "dataset" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "alerts_aggregated.parquet"
TEMP_DIR = PROJECT_ROOT / "temp"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

def sql_path(path):
    return path.resolve().as_posix().replace("'", "''")

connection = duckdb.connect()

connection.execute("SET memory_limit='3GB'")
connection.execute("SET threads=2")
connection.execute("SET preserve_insertion_order=false")
connection.execute(
    f"SET temp_directory='{sql_path(TEMP_DIR)}'"
)

print("Loading GUIDE evidence records...")

connection.execute(f"""
    CREATE OR REPLACE VIEW evidence AS
    SELECT *
    FROM read_csv_auto(
        '{sql_path(DATA_FILE)}',
        all_varchar=true,
        sample_size=20000,
        ignore_errors=false
    )
""")

total_rows = connection.execute(
    "SELECT COUNT(*) FROM evidence"
).fetchone()[0]

print(f"Evidence records: {total_rows:,}")
print("Aggregating records by organization and alert...")

aggregation_query = """
SELECT
    OrgId,
    AlertId,

    COUNT(*) AS evidence_count,

    MIN(TRY_CAST(NULLIF(TRIM(Timestamp), '') AS TIMESTAMPTZ))
        AS first_seen,

    MAX(TRY_CAST(NULLIF(TRIM(Timestamp), '') AS TIMESTAMPTZ))
        AS last_seen,

    string_agg(DISTINCT NULLIF(TRIM(IncidentId), ''), '|')
        AS incident_ids,

    string_agg(DISTINCT NULLIF(TRIM(AlertTitle), ''), '|')
        AS alert_titles,

    string_agg(DISTINCT NULLIF(TRIM(Category), ''), '|')
        AS categories,

    string_agg(DISTINCT NULLIF(TRIM(MitreTechniques), ''), '|')
        AS mitre_techniques,

    COUNT(DISTINCT NULLIF(TRIM(DetectorId), ''))
        AS unique_detectors,

    COUNT(DISTINCT NULLIF(TRIM(EntityType), ''))
        AS unique_entity_types,

    COUNT(DISTINCT NULLIF(TRIM(DeviceId), ''))
        AS unique_devices,

    COUNT(DISTINCT NULLIF(TRIM(IpAddress), ''))
        AS unique_ip_addresses,

    COUNT(DISTINCT NULLIF(TRIM(AccountSid), ''))
        AS unique_accounts,

    COUNT(DISTINCT NULLIF(TRIM(Sha256), ''))
        AS unique_file_hashes,

    COUNT(DISTINCT NULLIF(TRIM(Url), ''))
        AS unique_urls,

    string_agg(DISTINCT NULLIF(TRIM(IncidentGrade), ''), '|')
        AS incident_grades

FROM evidence
WHERE
    NULLIF(TRIM(OrgId), '') IS NOT NULL
    AND NULLIF(TRIM(AlertId), '') IS NOT NULL

GROUP BY OrgId, AlertId
"""

connection.execute(f"""
    COPY ({aggregation_query})
    TO '{sql_path(OUTPUT_FILE)}'
    (FORMAT PARQUET, COMPRESSION ZSTD)
""")

aggregated_rows = connection.execute(
    f"SELECT COUNT(*) FROM read_parquet('{sql_path(OUTPUT_FILE)}')"
).fetchone()[0]

print("\n" + "=" * 55)
print("ALERT AGGREGATION SUMMARY")
print("=" * 55)

print(f"Original evidence records: {total_rows:,}")
print(f"Aggregated alert records:  {aggregated_rows:,}")
print(f"Evidence records grouped:  {total_rows - aggregated_rows:,}")
print(f"Output file: {OUTPUT_FILE}")

print("\nSample aggregated records:")

sample = connection.execute(f"""
    SELECT
        OrgId,
        AlertId,
        evidence_count,
        categories,
        unique_devices,
        unique_ip_addresses
    FROM read_parquet('{sql_path(OUTPUT_FILE)}')
    LIMIT 5
""").fetchdf()

print(sample.to_string(index=False))

connection.close()

print("\nAlert aggregation completed successfully.")
