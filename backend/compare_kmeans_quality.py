
from pathlib import Path
import json
import time

import numpy as np
from scipy.sparse import load_npz
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ML_DIR = ROOT / "dataset" / "processed" / "ml"
REPORT_DIR = ROOT / "reports"

X = load_npz(ML_DIR / "alert_features_matrix.npz").tocsr()

baseline = pd.read_csv(ML_DIR / "kmeans_k3_assignments.csv")
baseline_labels = baseline["cluster_id"].to_numpy()

results = []

for n_init in [3, 10, 20]:
    print(f"\nTesting n_init={n_init}", flush=True)

    start = time.time()

    model = MiniBatchKMeans(
        n_clusters=3,
        random_state=42,
        batch_size=2048,
        n_init=n_init,
        max_iter=20,
        max_no_improvement=10,
        tol=0.0,
        reassignment_ratio=0.01
    )

    labels = model.fit_predict(X)

    silhouette = silhouette_score(
        X,
        labels,
        sample_size=5000,
        random_state=42
    )

    ari = adjusted_rand_score(
        baseline_labels,
        labels
    )

    result = {
        "n_init": n_init,
        "inertia": float(model.inertia_),
        "silhouette_sample": float(silhouette),
        "baseline_ari": float(ari),
        "execution_seconds": round(time.time() - start, 2)
    }

    results.append(result)

    print("Inertia:", result["inertia"])
    print("Silhouette:", round(silhouette, 4))
    print("Baseline ARI:", round(ari, 4))
    print("Time:", result["execution_seconds"])

REPORT_DIR.mkdir(parents=True, exist_ok=True)

with open(REPORT_DIR / "kmeans_quality_comparison.json", "w") as file:
    json.dump(results, file, indent=2)

print("\nQuality comparison complete.")
