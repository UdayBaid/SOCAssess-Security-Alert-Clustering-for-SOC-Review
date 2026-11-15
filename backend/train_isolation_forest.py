
import json
import time
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.sparse import load_npz, hstack
from sklearn.ensemble import IsolationForest

ROOT = Path(__file__).resolve().parent.parent
ML_DIR = ROOT / "dataset" / "processed" / "ml"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

start = time.time()

print("Loading feature matrix...")

X = load_npz(ML_DIR / "alert_features_matrix.npz")

with open(ML_DIR / "feature_names.json", "r") as f:
    feature_names = json.load(f)

if isinstance(feature_names, dict):
    feature_names = feature_names["feature_names"]

metadata = pd.read_csv(
    ML_DIR / "alert_feature_metadata.csv"
)

# Remove temporal features
excluded = {
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "is_weekend"
}

selected_indices = [
    i for i, name in enumerate(feature_names)
    if name not in excluded
]

X_reduced = X[:, selected_indices].tocsr()

print("Alerts:", X_reduced.shape[0])
print("Features:", X_reduced.shape[1])

print("\nTraining Isolation Forest...")

model = IsolationForest(
    n_estimators=200,
    max_samples=1024,
    contamination="auto",
    random_state=42,
    n_jobs=-1
)

predictions = model.fit_predict(X_reduced)
scores = model.decision_function(X_reduced)

results = metadata.copy()
results["anomaly_label"] = predictions
results["anomaly_score"] = scores

results["anomaly_status"] = np.where(
    predictions == -1,
    "Anomalous",
    "Normal"
)

print("\nANOMALY SUMMARY")

summary = results["anomaly_status"].value_counts()
print(summary)

print("\nAnomaly percentage:",
      round((predictions == -1).mean() * 100, 2))

results.to_csv(
    REPORT_DIR / "isolation_forest_assignments.csv",
    index=False
)

joblib.dump(
    model,
    MODEL_DIR / "isolation_forest.joblib"
)

report = {
    "total_alerts": int(len(results)),
    "features_used": int(X_reduced.shape[1]),
    "anomalies": int((predictions == -1).sum()),
    "normal_alerts": int((predictions == 1).sum()),
    "anomaly_percentage": round(
        float((predictions == -1).mean() * 100), 2
    ),
    "mean_anomaly_score": float(scores.mean()),
    "execution_seconds": round(time.time() - start, 2)
}

with open(
    REPORT_DIR / "isolation_forest_report.json", "w"
) as f:
    json.dump(report, f, indent=4)

print("\nModel saved.")
print("Assignments saved.")
print("Report saved.")
print("Execution time:", report["execution_seconds"])
