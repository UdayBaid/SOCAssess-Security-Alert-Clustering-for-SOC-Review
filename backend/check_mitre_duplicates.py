
from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "dataset" / "processed" / "alert_features.parquet"

con = duckdb.connect()
con.execute("SET memory_limit='2GB'")
con.execute("SET threads=2")

path = DATA_FILE.as_posix()

query = f"""
WITH technique_lists AS (
    SELECT
        AlertId,
        STRING_SPLIT(mitre_techniques, ';') AS techniques
    FROM read_parquet('{path}')
    WHERE mitre_techniques IS NOT NULL
      AND TRIM(mitre_techniques) <> ''
)
SELECT
    COUNT(*) AS alerts_with_mitre,
    COUNT(*) FILTER (
        WHERE ARRAY_LENGTH(techniques) >
              ARRAY_LENGTH(LIST_DISTINCT(techniques))
    ) AS alerts_with_duplicate_techniques
FROM technique_lists
"""

print(con.execute(query).df().to_string(index=False))

con.close()

