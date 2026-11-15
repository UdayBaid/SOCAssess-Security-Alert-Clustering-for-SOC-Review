
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parent.parent
ML_DIR = ROOT / "dataset" / "processed" / "ml"
REPORT_DIR = ROOT / "reports"

MATRIX_FILE = ML_DIR / "alert_features_matrix.npz"
ASSIGNMENTS_FILE = ML_DIR / "kmeans_k3_assignments.csv"
METADATA_FILE = ML_DIR / "alert_feature_metadata.csv"
REPORT_FILE = REPORT_DIR / "kmeans_stability.json"

print("=" * 60)
print("SOCAssess - K-MEANS CLUSTER STABILITY")
print("=" * 60)

X = load_npz(MATRIX_FILE).tocsr()

baseline = pd.read_csv(ASSIGNMENTS_FILE)
metadata = pd.read_csv(METADATA_FILE)

if X.shape[0] != len(baseline):
    raise ValueError("Matrix and baseline assignment row counts differ.")

if not baseline[["OrgId", "AlertId"]].equals(
    metadata[["OrgId", "AlertId"]]
):
    raise ValueError("Assignment and metadata row order differs.")

baseline_labels = baseline["cluster_id"].to_numpy()

print("Matrix shape:", X.shape)
print("Baseline clusters:", np.unique(baseline_labels))
print("Baseline seed: 42")

results = []

for seed in [7, 21, 100]:
    print(f"\nEvaluating random_state = {seed}", flush=True)

    start = time.time()

    model = MiniBatchKMeans(
        n_clusters=3,
        random_state=seed,
        batch_size=2048,
        n_init=3,
        max_iter=20,
        max_no_improvement=10,
        tol=0.0,
        reassignment_ratio=0.01
    )

    labels = model.fit_predict(X)

    ari = adjusted_rand_score(
        baseline_labels,
        labels
    )

    unique, counts = np.unique(labels, return_counts=True)

    cluster_sizes = {
        str(int(cluster)): int(count)
        for cluster, count in zip(unique, counts)
    }

    result = {
        "random_state": seed,
        "adjusted_rand_index": float(ari),
        "cluster_sizes": cluster_sizes,
        "iterations": int(model.n_iter_),
        "minibatch_steps": int(model.n_steps_),
        "execution_seconds": round(time.time() - start, 2)
    }

    results.append(result)

    print("ARI:", round(ari, 4))
    print("Cluster sizes:", cluster_sizes)
    print("Mini-batch steps:", model.n_steps_)
    print("Time:", result["execution_seconds"], "seconds")

REPORT_DIR.mkdir(parents=True, exist_ok=True)

with open(REPORT_FILE, "w") as file:
    json.dump(results, file, indent=2)

print("\n" + "=" * 60)
print("STABILITY VALIDATION COMPLETE")
print("Report:", REPORT_FILE)
print("=" * 60)
