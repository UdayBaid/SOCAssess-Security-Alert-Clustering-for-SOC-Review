
from pathlib import Path
import json
import time

import numpy as np
from scipy.sparse import load_npz
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parent.parent
ML_DIR = ROOT / "dataset" / "processed" / "ml"

MATRIX_FILE = ML_DIR / "alert_features_matrix.npz"
REPORT_FILE = ROOT / "reports" / "kmeans_evaluation.json"

print("=" * 60)
print("SOCAssess - K-MEANS BASELINE EVALUATION")
print("=" * 60)

X = load_npz(MATRIX_FILE).tocsr()

print("Matrix shape:", X.shape)
print("Matrix format:", X.format)
print("Matrix size (MB):", round(
    (X.data.nbytes + X.indices.nbytes + X.indptr.nbytes)
    / (1024 ** 2), 2
))

results = []

for k in [3, 4, 5, 6]:
    print(f"\nEvaluating K = {k}", flush=True)

    start = time.time()

    model = MiniBatchKMeans(
        n_clusters=k,
        random_state=42,
        batch_size=2048,
        n_init=3,
        max_iter=20,
        max_no_improvement=10,
        tol=0.0,
        reassignment_ratio=0.01
    )

    labels = model.fit_predict(X)

    score = silhouette_score(
        X,
        labels,
        sample_size=5000,
        random_state=42,
        metric="euclidean"
    )

    unique, counts = np.unique(labels, return_counts=True)

    cluster_sizes = {
        int(cluster): int(count)
        for cluster, count in zip(unique, counts)
    }

    result = {
        "k": k,
        "inertia": float(model.inertia_),
        "silhouette_sample": float(score),
        "iterations": int(model.n_iter_),
        "cluster_sizes": cluster_sizes,
        "execution_seconds": round(time.time() - start, 2),
        "minibatch_steps": int(model.n_steps_)
    }

    results.append(result)

    print("Inertia:", round(model.inertia_, 2))
    print("Silhouette:", round(score, 4))
    print("Iterations:", model.n_iter_)
    print("Cluster sizes:", cluster_sizes)
    print("Time:", result["execution_seconds"], "seconds")
    print("Mini-batch steps:", model.n_steps_)

REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

with open(REPORT_FILE, "w") as file:
    json.dump(results, file, indent=2)

print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("Report:", REPORT_FILE)
print("=" * 60)
