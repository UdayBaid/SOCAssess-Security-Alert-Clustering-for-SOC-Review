
from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parent.parent

SOURCE = ROOT / "dataset" / "GUIDE_Test.csv"
AGGREGATED = ROOT / "dataset" / "processed" / "alerts_aggregated.parquet"

con = duckdb.connect()

def safe_path(path):
    return path.resolve().as_posix().replace("'", "''")

if not SOURCE.exists():
    raise FileNotFoundError(SOURCE)

if not AGGREGATED.exists():
    raise FileNotFoundError(AGGREGATED)

source_path = safe_path(SOURCE)
output_path = safe_path(AGGREGATED)

con.execute(f"""
    CREATE VIEW source_data AS
    SELECT *
    FROM read_csv_auto(
        '{source_path}',
        all_varchar=true,
        sample_size=20000
    )
""")

con.execute(f"""
    CREATE VIEW aggregated_data AS
    SELECT *
    FROM read_parquet('{output_path}')
""")

source_rows = con.execute("""
    SELECT COUNT(*) FROM source_data
""").fetchone()[0]

valid_source_rows = con.execute("""
    SELECT COUNT(*)
    FROM source_data
    WHERE NULLIF(TRIM(OrgId), '') IS NOT NULL
      AND NULLIF(TRIM(AlertId), '') IS NOT NULL
""").fetchone()[0]

invalid_key_rows = source_rows - valid_source_rows

aggregated_rows = con.execute("""
    SELECT COUNT(*) FROM aggregated_data
""").fetchone()[0]

evidence_sum = con.execute("""
    SELECT SUM(evidence_count) FROM aggregated_data
""").fetchone()[0]

stats = con.execute("""
    SELECT
        MIN(evidence_count),
        ROUND(AVG(evidence_count), 2),
        MEDIAN(evidence_count),
        QUANTILE_CONT(evidence_count, 0.95),
        MAX(evidence_count)
    FROM aggregated_data
""").fetchone()

print("\n" + "=" * 55)
print("AGGREGATION INTEGRITY REPORT")
print("=" * 55)

print(f"Original evidence rows:       {source_rows:,}")
print(f"Rows with valid alert keys:   {valid_source_rows:,}")
print(f"Rows with missing alert keys: {invalid_key_rows:,}")
print(f"Aggregated alert groups:      {aggregated_rows:,}")
print(f"Sum of evidence_count:        {evidence_sum:,}")

print("\nEvidence count per alert:")
print(f"Minimum:  {stats[0]}")
print(f"Average:  {stats[1]}")
print(f"Median:   {stats[2]}")
print(f"95th percentile: {stats[3]}")
print(f"Maximum:  {stats[4]}")

print("\nIntegrity check:")
if evidence_sum == valid_source_rows:
    print("PASS: Every row with valid alert keys is accounted for.")
else:
    print("FAIL: Evidence row totals do not match.")

if aggregated_rows == 637320:
    print("PASS: Aggregated alert count matches the validation result.")
else:
    print("CHECK: Aggregated count differs from the earlier result.")

con.close()
