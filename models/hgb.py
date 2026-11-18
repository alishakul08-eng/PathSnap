
import os
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from evaluate_models import cross_validate_model

# Load training data
df = pd.read_csv("train_candidates_final.csv")

print("Dataset:", df.shape)
print("Trips:", df["trip_id"].nunique())

# Improved HGB configuration
model = HistGradientBoostingClassifier(
    max_iter=250,
    learning_rate=0.05,
    max_leaf_nodes=63,
    min_samples_leaf=30,
    l2_regularization=1.0,
    early_stopping=False,
    random_state=42
)

# Evaluate using shared GroupKFold
results = cross_validate_model(
    "HistGradientBoosting",
    model,
    df,
    n_splits=5
)

os.makedirs("evaluation/results", exist_ok=True)

results.to_csv(
    "evaluation/results/HistGradientBoosting_cv.csv",
    index=False
)

print("\nHGB RESULTS:")
print(results)

print("\nAVERAGE ACCURACY:",
      results["accuracy"].mean())

print("\nAVERAGE MACRO F1:",
      results["macro_f1"].mean())
