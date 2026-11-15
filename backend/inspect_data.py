from pathlib import Path
import csv
import json
import duckdb
import os

REPORT_DIR = Path(os.environ.get("SOCA_REPORT_DIR", "reports"))
DATA_FILE = Path(os.environ.get("DATA_FILE_PATH", "dataset/GUIDE_Test.csv"))
REPORT_DIR = Path(r"C:\Users\udot\OneDrive\Desktop\New folder\SOCAassess\reports")
SAMPLE_SIZE = 100_000   

REPORT_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

file_size_mb = DATA_FILE.stat().st_size / (1024 ** 2)

print("=" * 60)
print("SOCAssess - Dataset Inspection")
print("=" * 60)
print(f"\nFile: {DATA_FILE.name}")
print(f"Size: {file_size_mb:.2f} MB")

with DATA_FILE.open("r", encoding="utf-8-sig", newline="") as f:
    reader = csv.reader(f)
    header = next(reader)
    preview = [row for _, row in zip(range(5), reader)]

print(f"\nNumber of columns: {len(header)}")
print("\nColumn names:")
for i, col in enumerate(header, start=1):
    print(f"{i:02d}. {col}")

print("\nPreview:")
for row in preview:
    print(row)

escaped_path = DATA_FILE.as_posix().replace("'", "''")
source = f"read_csv_auto('{escaped_path}', sample_size={SAMPLE_SIZE})"

connection = duckdb.connect()
try:
    schema_rows = connection.execute(f"DESCRIBE SELECT * FROM {source}").fetchall()
    schema = [
        {"column_name": r[0], "column_type": r[1], "null": r[2]}
        for r in schema_rows
    ]
    row_count = connection.execute(f"SELECT COUNT(*) FROM {source}").fetchone()[0]
finally:
    connection.close()

print(f"\nRow count: {row_count:,}")
print("\nDetected schema:")
for col in schema:
    print(f"  {col['column_name']:<35} {col['column_type']}")

report = {
    "file_name": DATA_FILE.name,
    "file_size_mb": round(file_size_mb, 2),
    "number_of_columns": len(header),
    "row_count": row_count,
    "columns": header,
    "preview_rows": preview,
    "schema": schema,
}

report_path = REPORT_DIR / "dataset_inspection.json"
with report_path.open("w", encoding="utf-8") as f:
    json.dump(report, f, indent=4, default=str)

print("\nInspection completed.")
print(f"Report saved at: {report_path}")