
from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "dataset" / "processed" / "alert_features.parquet"

con = duckdb.connect()
con.execute("SET memory_limit='2GB'")
con.execute("SET threads=2")

path = DATA_FILE.as_posix()

query = f"""
WITH split_techniques AS (
    SELECT DISTINCT
        TRIM(technique) AS technique
    FROM read_parquet('{path}'),
    UNNEST(STRING_SPLIT(mitre_techniques, ';')) AS t(technique)
    WHERE mitre_techniques IS NOT NULL
      AND TRIM(mitre_techniques) <> ''
)
SELECT
    technique,
    COUNT(*) AS occurrence
FROM (
    SELECT
        TRIM(technique) AS technique
    FROM read_parquet('{path}'),
    UNNEST(STRING_SPLIT(mitre_techniques, ';')) AS t(technique)
    WHERE mitre_techniques IS NOT NULL
      AND TRIM(mitre_techniques) <> ''
)
GROUP BY technique
ORDER BY occurrence DESC
"""

result = con.execute(query).df()

print("TOTAL UNIQUE INDIVIDUAL MITRE TECHNIQUES:",
      result["technique"].nunique())

print("\nTOP 30 MITRE TECHNIQUES")
print(result.head(30).to_string(index=False))

print("\nTECHNIQUE COUNT DISTRIBUTION")
print(result["occurrence"].describe())

con.close()
