
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

rng = np.random.default_rng(42)
indices = np.sort(
    rng.choice(X.shape[0], size=5000, replace=False)
)

model = DBSCAN(
    eps=0.08,
    min_samples=5,
    metric="cosine",
    algorithm="brute",
    n_jobs=1
)

labels = model.fit_predict(X[indices])

sample = metadata.iloc[indices].copy()
sample["cluster_id"] = labels

for df in [sample, features]:
    df["OrgId"] = pd.to_numeric(df["OrgId"]).astype("int64")
    df["AlertId"] = pd.to_numeric(df["AlertId"]).astype("int64")

data = sample.merge(
    features[
        [
            "OrgId",
            "AlertId",
            "evidence_count",
            "unique_ip_addresses",
            "categories",
            "alert_titles",
            "mitre_techniques"
        ]
    ],
    on=["OrgId", "AlertId"],
    how="left",
    validate="one_to_one"
)

selected_clusters = [0, 3, 16, 19, 20, 24, 15]

for cluster_id in selected_clusters:
    group = data[data["cluster_id"] == cluster_id].copy()

    print(f"\n{'=' * 55}")
    print(f"CLUSTER {cluster_id} | ALERTS: {len(group)}")
    print("=" * 55)

    titles = (
        group.assign(
            title=group["alert_titles"].str.split(";")
        )
        .explode("title")
        .dropna(subset=["title"])
        .groupby("title")
        .size()
        .sort_values(ascending=False)
        .head(10)
    )

    print("\nTOP ALERT TITLES")
    print(titles.to_string())

    print("\nTOP MITRE TECHNIQUES")
    techniques = (
        group.assign(
            technique=group["mitre_techniques"].str.split(";")
        )
        .explode("technique")
        .dropna(subset=["technique"])
        .groupby("technique")
        .size()
        .sort_values(ascending=False)
        .head(10)
    )

    print(techniques.to_string())

    print("\nUNIQUE ALERT TITLES:", group["alert_titles"].nunique())

    if cluster_id == 15:
        print("\nHIGH IP COUNT ALERTS")
        print(
            group[
                [
                    "OrgId",
                    "AlertId",
                    "evidence_count",
                    "unique_ip_addresses",
                    "categories",
                    "alert_titles"
                ]
            ].sort_values(
                "unique_ip_addresses",
                ascending=False
            ).to_string(index=False)
        )

REPORT_DIR.mkdir(parents=True, exist_ok=True)

data.to_csv(
    REPORT_DIR / "dbscan_sample_assignments_with_details.csv",
    index=False
)

print("\nDetailed inspection complete.")
