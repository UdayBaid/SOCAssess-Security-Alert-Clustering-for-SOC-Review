
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.cluster import DBSCAN

ROOT = Path(__file__).resolve().parent.parent
ML_DIR = ROOT / "dataset" / "processed" / "ml"
PROCESSED_DIR = ROOT / "dataset" / "processed"
REPORT_DIR = ROOT / "reports"

X = load_npz(ML_DIR / "alert_features_matrix.npz").tocsr()
metadata = pd.read_csv(ML_DIR / "alert_feature_metadata.csv")
features = pd.read_parquet(PROCESSED_DIR / "alert_features.parquet")

SAMPLE_SIZE = 5000
RANDOM_STATE = 42

rng = np.random.default_rng(RANDOM_STATE)
indices = np.sort(
    rng.choice(X.shape[0], size=SAMPLE_SIZE, replace=False)
)

X_sample = X[indices]
sample_metadata = metadata.iloc[indices].copy()

model = DBSCAN(
    eps=0.08,
    min_samples=5,
    metric="cosine",
    algorithm="brute",
    n_jobs=1
)

print("Running DBSCAN...", flush=True)
labels = model.fit_predict(X_sample)

sample_metadata["cluster_id"] = labels

for df in [sample_metadata, features]:
    df["OrgId"] = pd.to_numeric(
        df["OrgId"], errors="raise"
    ).astype("int64")

    df["AlertId"] = pd.to_numeric(
        df["AlertId"], errors="raise"
    ).astype("int64")
data = sample_metadata.merge(
    features[
        [
            "OrgId",
            "AlertId",
            "evidence_count",
            "unique_devices",
            "unique_ip_addresses",
            "unique_accounts",
            "categories",
            "mitre_techniques"
        ]
    ],
    on=["OrgId", "AlertId"],
    how="left",
    validate="one_to_one"
)

cluster_data = data[data["cluster_id"] != -1].copy()

summary = (
    data.groupby("cluster_id")
    .agg(
        alert_count=("AlertId", "size"),
        mean_evidence=("evidence_count", "mean"),
        median_evidence=("evidence_count", "median"),
        mean_devices=("unique_devices", "mean"),
        mean_ip_addresses=("unique_ip_addresses", "mean"),
        mean_accounts=("unique_accounts", "mean")
    )
    .reset_index()
)

summary["percentage"] = (
    summary["alert_count"] / SAMPLE_SIZE * 100
)

category_data = (
    cluster_data.assign(
        category=cluster_data["categories"].str.split(";")
    )
    .explode("category")
)

category_summary = (
    category_data.groupby(["cluster_id", "category"])
    .size()
    .reset_index(name="alert_occurrences")
    .sort_values(
        ["cluster_id", "alert_occurrences"],
        ascending=[True, False]
    )
)

mitre_data = (
    cluster_data.dropna(subset=["mitre_techniques"])
    .assign(
        technique=cluster_data["mitre_techniques"].str.split(";")
    )
    .explode("technique")
)

mitre_summary = (
    mitre_data.groupby(["cluster_id", "technique"])
    .size()
    .reset_index(name="alert_occurrences")
    .sort_values(
        ["cluster_id", "alert_occurrences"],
        ascending=[True, False]
    )
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)

summary.to_csv(
    REPORT_DIR / "dbscan_cluster_summary.csv",
    index=False
)

category_summary.to_csv(
    REPORT_DIR / "dbscan_cluster_categories.csv",
    index=False
)

mitre_summary.to_csv(
    REPORT_DIR / "dbscan_cluster_mitre.csv",
    index=False
)

print("\nCLUSTER SUMMARY")
print(summary.sort_values("alert_count", ascending=False).to_string(index=False))

print("\nTOP CATEGORIES")
print(category_summary.groupby("cluster_id").head(3).to_string(index=False))

print("\nTOP MITRE TECHNIQUES")
print(mitre_summary.groupby("cluster_id").head(3).to_string(index=False))

print("\nReports saved in:", REPORT_DIR)
