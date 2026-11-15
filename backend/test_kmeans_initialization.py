
from pathlib import Path
import json
import time

import numpy as np
from scipy.sparse import load_npz
from sklearn.cluster import MiniBatchKMeans

ROOT = Path(__file__).resolve().parent.parent
ML_DIR = ROOT / "dataset" / "processed" / "ml"
REPORT_DIR = ROOT / "reports"

X = load_npz(ML_DIR / "alert_features_matrix.npz").tocsr()

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

    unique, counts = np.unique(labels, return_counts=True)

    cluster_sizes = {
        str(int(cluster)): int(count)
        for cluster, count in zip(unique, counts)
    }

    result = {
        "n_init": n_init,
        "inertia": float(model.inertia_),
        "iterations": int(model.n_iter_),
        "minibatch_steps": int(model.n_steps_),
        "cluster_sizes": cluster_sizes,
        "execution_seconds": round(time.time() - start, 2)
    }

    results.append(result)

    print("Inertia:", model.inertia_)
    print("Cluster sizes:", cluster_sizes)
    print("Mini-batch steps:", model.n_steps_)
    print("Time:", result["execution_seconds"])

REPORT_DIR.mkdir(parents=True, exist_ok=True)

with open(REPORT_DIR / "kmeans_initialization_test.json", "w") as file:
    json.dump(results, file, indent=2)

print("\nInitialization experiment complete.")
