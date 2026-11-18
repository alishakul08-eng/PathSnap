import sys
import pandas as pd
from models_config import get_models
from evaluate_models import cross_validate_model

MODEL_NAME = "RandomForest"
QUICK = "--quick" in sys.argv   # 20 trips only, for a fast pipeline test

df = pd.read_csv("data/processed/train_candidates_final.csv")
if QUICK:
    keep = sorted(df.trip_id.unique())[:20]
    df = df[df.trip_id.isin(keep)].reset_index(drop=True)
print("rows:", len(df), "| trips:", df.trip_id.nunique(), "| quick:", QUICK)

results = cross_validate_model(MODEL_NAME, get_models()[MODEL_NAME], df)
out = f"evaluation/results/{MODEL_NAME}_cv{'_quick' if QUICK else ''}.csv"
results.to_csv(out, index=False)
print(results.drop(columns="fold").groupby("model").mean(numeric_only=True).round(4).T)
