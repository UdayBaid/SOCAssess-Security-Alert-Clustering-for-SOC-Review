
from pathlib import Path
import pandas as pd
from scipy.sparse import load_npz

ROOT = Path(__file__).resolve().parent.parent
ML_DIR = ROOT / "dataset" / "processed" / "ml"

X = load_npz(ML_DIR / "alert_features_matrix.npz")
metadata = pd.read_csv(ML_DIR / "alert_feature_metadata.csv", nrows=5)

print("Feature matrix rows:", X.shape[0])
print("Metadata columns:")
print(metadata.columns.tolist())

full_metadata = pd.read_csv(
    ML_DIR / "alert_feature_metadata.csv",
    usecols=["OrgId", "AlertId"]
)

print("\nMetadata rows:", len(full_metadata))
print("Unique alert keys:", full_metadata.drop_duplicates().shape[0])
print("\nSample:")
print(metadata.head())
