
from pathlib import Path
import json
import duckdb
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = PROJECT_ROOT / "dataset" / "GUIDE_Test.csv"

REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TEMP_DIR = PROJECT_ROOT / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

connection = duckdb.connect()


connection.execute("SET memory_limit = '3GB'")
connection.execute("SET threads = 2")
connection.execute("SET preserve_insertion_order = false")
connection.execute(
    f"SET temp_directory = '{TEMP_DIR.as_posix()}'"
)

file_path = DATA_FILE.as_posix().replace("'", "''")

print("=" * 60)
print("SOCAssess - Data Validation")
print("=" * 60)
print("\nReading dataset...")

connection.execute(f"""
    CREATE VIEW guide_data AS
    SELECT *
    FROM read_csv_auto(
        '{file_path}',
        header = true,
        all_varchar = true,
        sample_size = 20000,
        ignore_errors = false
    )
""")

columns = connection.execute("""
    DESCRIBE SELECT * FROM guide_data
""").fetchall()

column_names = [column[0] for column in columns]

print(f"Detected columns: {len(column_names)}")


missing_checks = []

for column in column_names:
    safe_column = '"' + column.replace('"', '""') + '"'

    missing_checks.append(
        f"""COUNT(*) FILTER (
            WHERE NULLIF(TRIM({safe_column}), '') IS NULL
        ) AS "{column}" """
    )

missing_query = ",\n".join(missing_checks)


metrics_query = f"""
SELECT
    COUNT(*) AS total_rows,

    COUNT(DISTINCT NULLIF(TRIM("Id"), ''))
        AS unique_record_ids,

    COUNT(*) FILTER (
        WHERE NULLIF(TRIM("Id"), '') IS NULL
    ) AS missing_record_ids,

    COUNT(*) FILTER (
        WHERE NULLIF(TRIM("Timestamp"), '') IS NOT NULL
        AND TRY_CAST("Timestamp" AS TIMESTAMPTZ) IS NULL
    ) AS invalid_timestamps,

    MIN(TRY_CAST("Timestamp" AS TIMESTAMPTZ))
        AS earliest_timestamp,

    MAX(TRY_CAST("Timestamp" AS TIMESTAMPTZ))
        AS latest_timestamp,

    COUNT(DISTINCT NULLIF(TRIM("AlertId"), ''))
        AS unique_alert_ids,

    COUNT(DISTINCT NULLIF(TRIM("IncidentId"), ''))
        AS unique_incident_ids,

    COUNT(DISTINCT NULLIF(TRIM("OrgId"), ''))
        AS unique_organizations,

    {missing_query}

FROM guide_data
"""

print("\nRunning validation checks...")
metrics = connection.execute(metrics_query).fetchone()
metric_names = [
    description[0]
    for description in connection.description
]

results = dict(zip(metric_names, metrics))

total_rows = results["total_rows"]
unique_ids = results["unique_record_ids"]

results["duplicate_record_ids"] = (
    total_rows
    - results["missing_record_ids"]
    - unique_ids
)


missing_values = {
    column: results.pop(column)
    for column in column_names
}

# Check incident-grade distribution.
grade_rows = connection.execute("""
    SELECT
        COALESCE(
            NULLIF(TRIM("IncidentGrade"), ''),
            '<MISSING>'
        ) AS incident_grade,
        COUNT(*) AS record_count
    FROM guide_data
    GROUP BY 1
    ORDER BY record_count DESC
""").fetchall()

grade_distribution = {
    grade: count for grade, count in grade_rows
}


category_rows = connection.execute("""
    SELECT
        COALESCE(
            NULLIF(TRIM("Category"), ''),
            '<MISSING>'
        ) AS category,
        COUNT(*) AS record_count
    FROM guide_data
    GROUP BY 1
    ORDER BY record_count DESC
    LIMIT 20
""").fetchall()

category_distribution = {
    category: count for category, count in category_rows
}

report = {
    "dataset": DATA_FILE.name,
    "file_size_mb": round(
        DATA_FILE.stat().st_size / (1024 ** 2), 2
    ),
    "column_count": len(column_names),
    "columns": column_names,
    "metrics": results,
    "missing_values": missing_values,
    "incident_grade_distribution": grade_distribution,
    "top_20_categories": category_distribution
}

report_path = REPORT_DIR / "validation_report.json"

with report_path.open("w", encoding="utf-8") as file:
    json.dump(report, file, indent=4, default=str)

connection.close()

print("\n" + "=" * 60)
print("VALIDATION SUMMARY")
print("=" * 60)

print(f"Total rows: {total_rows:,}")
print(f"Unique record IDs: {unique_ids:,}")
print(f"Duplicate record IDs: {results['duplicate_record_ids']:,}")
print(f"Missing record IDs: {results['missing_record_ids']:,}")
print(f"Invalid timestamps: {results['invalid_timestamps']:,}")
print(f"Unique alerts: {results['unique_alert_ids']:,}")
print(f"Unique incidents: {results['unique_incident_ids']:,}")
print(f"Unique organizations: {results['unique_organizations']:,}")

print("\nMissing values by column:")
for column, count in missing_values.items():
    if count:
        percentage = (count / total_rows) * 100
        print(f"{column}: {count:,} ({percentage:.2f}%)")

print("\nIncident grade distribution:")
for grade, count in grade_distribution.items():
    print(f"{grade}: {count:,}")

print(f"\nReport saved at: {report_path}")
print("\nData validation completed.")
