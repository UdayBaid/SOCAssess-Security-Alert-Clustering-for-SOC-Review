
from pathlib import Path
import json
import time

import joblib
import pandas as pd
from scipy.sparse import load_npz
from sklearn.cluster import MiniBatchKMeans

ROOT = Path(__file__).resolve().parent.parent
ML_DIR = ROOT / "dataset" / "processed" / "ml"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MATRIX_FILE = ML_DIR / "alert_features_matrix.npz"
METADATA_FILE = ML_DIR / "alert_feature_metadata.csv"

MODEL_FILE = MODEL_DIR / "kmeans_k3.joblib"
ASSIGNMENTS_FILE = ML_DIR / "kmeans_k3_assignments.csv"
REPORT_FILE = REPORT_DIR / "kmeans_k3_training.json"

print("=" * 60)
print("SOCAssess - K=3 MODEL TRAINING")
print("=" * 60)

X = load_npz(MATRIX_FILE).tocsr()
metadata = pd.read_csv(METADATA_FILE)

if X.shape[0] != len(metadata):
    raise ValueError("Feature matrix and metadata row counts do not match.")

if metadata.duplicated(["OrgId", "AlertId"]).any():
    raise ValueError("Duplicate alert keys found in metadata.")

print("Alerts:", X.shape[0])
print("Features:", X.shape[1])

model = MiniBatchKMeans(
    n_clusters=3,
    random_state=42,
    batch_size=2048,
    n_init=3,
    max_iter=20,
    max_no_improvement=10,
    tol=0.0,
    reassignment_ratio=0.01
)

print("\nTraining K-Means...")
start = time.time()

labels = model.fit_predict(X)

training_seconds = round(time.time() - start, 2)

metadata["cluster_id"] = labels
metadata.to_csv(ASSIGNMENTS_FILE, index=False)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(model, MODEL_FILE)

cluster_sizes = {
    str(int(cluster)): int(count)
    for cluster, count in zip(*__import__("numpy").unique(labels, return_counts=True))
}

report = {
    "algorithm": "MiniBatchKMeans",
    "n_clusters": 3,
    "random_state": 42,
    "alerts": int(X.shape[0]),
    "features": int(X.shape[1]),
    "inertia": float(model.inertia_),
    "iterations": int(model.n_iter_),
    "minibatch_steps": int(model.n_steps_),
    "cluster_sizes": cluster_sizes,
    "training_seconds": training_seconds
}

with open(REPORT_FILE, "w") as file:
    json.dump(report, file, indent=2)

print("\nTraining complete.")
print("Cluster sizes:", cluster_sizes)
print("Mini-batch steps:", model.n_steps_)
print("Training time:", training_seconds, "seconds")
print("Model saved:", MODEL_FILE)
print("Assignments saved:", ASSIGNMENTS_FILE)
print("Report saved:", REPORT_FILE)
