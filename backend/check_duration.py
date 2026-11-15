
from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parent.parent
PARQUET_FILE = ROOT / "dataset" / "processed" / "alert_features.parquet"

con = duckdb.connect()
con.execute("SET memory_limit='2GB'")
con.execute("SET threads=2")

query = f"""
SELECT
    COUNT(*) AS total_alerts,
    COUNT(*) FILTER (
        WHERE first_seen = last_seen
    ) AS same_timestamp_alerts,
    COUNT(*) FILTER (
        WHERE first_seen < last_seen
    ) AS different_timestamp_alerts,
    MIN(epoch(last_seen) - epoch(first_seen)) AS min_duration,
    MAX(epoch(last_seen) - epoch(first_seen)) AS max_duration,
    AVG(epoch(last_seen) - epoch(first_seen)) AS avg_duration
FROM read_parquet('{PARQUET_FILE.as_posix()}')
"""

print("DURATION SUMMARY")
print(con.execute(query).df().to_string(index=False))

print("\nSAMPLE ALERTS WITH DIFFERENT TIMESTAMPS")
sample_query = f"""
SELECT
    AlertId,
    first_seen,
    last_seen,
    epoch(last_seen) - epoch(first_seen) AS actual_duration
FROM read_parquet('{PARQUET_FILE.as_posix()}')
WHERE first_seen < last_seen
LIMIT 10
"""

print(con.execute(sample_query).df().to_string(index=False))
con.close()
