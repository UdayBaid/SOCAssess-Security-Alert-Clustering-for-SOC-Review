
from pathlib import Path
import duckdb
import json

ROOT = Path(__file__).resolve().parent.parent

INPUT = ROOT / "dataset" / "processed" / "alert_features.parquet"
REPORT_DIR = ROOT / "reports"
REPORT_FILE = REPORT_DIR / "feature_profile.json"
TEMP_DIR = ROOT / "temp"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

def safe_path(path):
    return path.resolve().as_posix().replace("'", "''")

def quote_identifier(name):
    return '"' + name.replace('"', '""') + '"'

if not INPUT.exists():
    raise FileNotFoundError(INPUT)

con = duckdb.connect()

con.execute("SET memory_limit='3GB'")
con.execute("SET threads=2")
con.execute("SET preserve_insertion_order=false")
con.execute(f"SET temp_directory='{safe_path(TEMP_DIR)}'")

con.execute(f"""
    CREATE VIEW features AS
    SELECT * FROM read_parquet('{safe_path(INPUT)}')
""")

schema = con.execute(
    "DESCRIBE SELECT * FROM features"
).fetchall()

total_rows = con.execute(
    "SELECT COUNT(*) FROM features"
).fetchone()[0]

numeric_types = (
    "TINYINT", "SMALLINT", "INTEGER", "BIGINT",
    "HUGEINT", "UTINYINT", "USMALLINT", "UINTEGER",
    "UBIGINT", "FLOAT", "DOUBLE", "DECIMAL"
)

expressions = []
metadata = []

position = 0

for column_name, data_type, *_ in schema:
    col = quote_identifier(column_name)
    dtype = data_type.upper()

    numeric = any(dtype.startswith(t) for t in numeric_types)

    item = {
        "column": column_name,
        "data_type": data_type,
        "numeric": numeric,
        "null_pos": position,
        "distinct_pos": position + 1
    }

    expressions.append(
        f"COUNT(*) FILTER (WHERE {col} IS NULL)"
    )
    expressions.append(
        f"APPROX_COUNT_DISTINCT({col})"
    )

    position += 2

    if numeric:
        item["stats_pos"] = position

        expressions.extend([
            f"MIN({col})",
            f"MAX({col})",
            f"AVG({col})",
            f"STDDEV_POP({col})",
            f"APPROX_QUANTILE({col}, 0.95)",
            f"APPROX_QUANTILE({col}, 0.99)"
        ])

        position += 6

    metadata.append(item)

print("Running corrected feature profiling...")

result = con.execute(
    f"SELECT {', '.join(expressions)} FROM features"
).fetchone()

profile = []

for item in metadata:
    null_count = int(result[item["null_pos"]])
    unique_count = int(result[item["distinct_pos"]])

    row = {
        "column": item["column"],
        "data_type": item["data_type"],
        "numeric": item["numeric"],
        "missing_count": null_count,
        "missing_percentage": round(
            null_count / total_rows * 100, 4
        ),
        "approx_unique_values": unique_count
    }

    if item["numeric"]:
        p = item["stats_pos"]

        row.update({
            "min": result[p],
            "max": result[p + 1],
            "mean": result[p + 2],
            "std_dev": result[p + 3],
            "p95": result[p + 4],
            "p99": result[p + 5]
        })

    profile.append(row)

report = {
    "dataset": INPUT.name,
    "total_alerts": total_rows,
    "total_columns": len(metadata),
    "numeric_columns": sum(x["numeric"] for x in metadata),
    "columns": profile
}

with open(REPORT_FILE, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=4, default=str)

print("\n" + "=" * 55)
print("CORRECTED FEATURE PROFILE")
print("=" * 55)

print(f"Total alerts: {total_rows:,}")
print(f"Total columns: {len(metadata)}")

for item in profile:
    print(f"\n{item['column']} ({item['data_type']})")
    print(f"  Missing: {item['missing_percentage']}%")
    print(f"  Unique: {item['approx_unique_values']:,}")

    if item["numeric"]:
        print(f"  Min: {item['min']}")
        print(f"  Max: {item['max']}")
        print(f"  Mean: {item['mean']}")
        print(f"  P95: {item['p95']}")
        print(f"  P99: {item['p99']}")

print(f"\nReport saved: {REPORT_FILE}")

con.close()
print("\nCorrected profiling completed.")
