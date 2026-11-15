
from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "dataset" / "processed" / "alert_features.parquet"

con = duckdb.connect()
con.execute("SET memory_limit='2GB'")
con.execute("SET threads=2")

file_path = DATA_FILE.as_posix()

queries = {
    "TOP CATEGORIES": f"""
        SELECT categories, COUNT(*) AS alert_count
        FROM read_parquet('{file_path}')
        GROUP BY categories
        ORDER BY alert_count DESC
        LIMIT 25
    """,
    "TOP ALERT TITLES": f"""
        SELECT alert_titles, COUNT(*) AS alert_count
        FROM read_parquet('{file_path}')
        GROUP BY alert_titles
        ORDER BY alert_count DESC
        LIMIT 20
    """,
    "MITRE TECHNIQUE SUMMARY": f"""
        SELECT
            COUNT(*) AS total_alerts,
            COUNT(*) FILTER (
                WHERE mitre_techniques IS NULL
                OR TRIM(mitre_techniques) = ''
            ) AS missing_mitre,
            COUNT(*) FILTER (
                WHERE mitre_techniques IS NOT NULL
                AND TRIM(mitre_techniques) <> ''
            ) AS alerts_with_mitre,
            MAX(LENGTH(mitre_techniques)) AS max_text_length
        FROM read_parquet('{file_path}')
    """,
    "MITRE TECHNIQUE SAMPLES": f"""
        SELECT mitre_techniques, COUNT(*) AS alert_count
        FROM read_parquet('{file_path}')
        WHERE mitre_techniques IS NOT NULL
          AND TRIM(mitre_techniques) <> ''
        GROUP BY mitre_techniques
        ORDER BY alert_count DESC
        LIMIT 20
    """,
    "CATEGORICAL COLUMN COUNTS": f"""
        SELECT
            COUNT(DISTINCT categories) AS unique_categories,
            COUNT(DISTINCT alert_titles) AS unique_alert_titles,
            COUNT(DISTINCT mitre_techniques) AS unique_mitre_combinations
        FROM read_parquet('{file_path}')
    """
}

for title, query in queries.items():
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)
    print(con.execute(query).df().to_string(index=False))

con.close()
