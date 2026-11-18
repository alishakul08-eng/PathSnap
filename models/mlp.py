
import os
import time
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, log_loss
)

# Load dataset
df = pd.read_csv("train_candidates_final.csv")

FEATURES = [
    "distance_m", "heading_difference", "candidate_rank",
    "arc_len_m", "proj_frac", "speed_mps", "dt_prev",
    "heading_change_deg", "dist_prev", "dist_next",
    "dist_min_obs", "dist_ratio_best", "has_reverse_twin",
    "same_arc_as_prev_best", "connects_from_prev_best",
    "arc_persistence"
]

X = df[FEATURES]
y = df["label"]
groups = df["trip_id"]

cv = GroupKFold(n_splits=5)
results = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups), 1):

    print(f"\nStarting MLP Fold {fold}/5...", flush=True)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            batch_size=1024,
            max_iter=100,
            early_stopping=False,
            random_state=42
        ))
    ])

    start = time.time()
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    train_time = time.time() - start

    start = time.time()
    probabilities = model.predict_proba(X.iloc[val_idx])
    inference_time = time.time() - start

    positive_index = list(model.classes_).index(1)
    proba = probabilities[:, positive_index]
    predictions = (proba >= 0.5).astype(int)
    actual = y.iloc[val_idx]

    results.append({
        "model": "MLP",
        "fold": fold,
        "accuracy": accuracy_score(actual, predictions),
        "macro_f1": f1_score(actual, predictions, average="macro"),
        "precision": precision_score(actual, predictions, zero_division=0),
        "recall": recall_score(actual, predictions, zero_division=0),
        "log_loss": log_loss(actual, probabilities, labels=model.classes_),
        "train_time_s": train_time,
        "infer_time_s": inference_time
    })

    print("Fold completed!", flush=True)
    print(results[-1], flush=True)

os.makedirs("evaluation/results", exist_ok=True)

pd.DataFrame(results).to_csv(
    "evaluation/results/MLP_cv.csv", index=False
)

print("\nMLP RESULTS SAVED!")
print("Number of trips:", df["trip_id"].nunique())
print("Total rows:", len(df))
print("Columns:", df.columns.tolist())