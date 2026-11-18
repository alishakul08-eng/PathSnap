# models/check_student_test.py  (development comparison only)
import time
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, log_loss
from models_config import get_models
from evaluate_models import FEATURES, top1_accuracy

MODEL_NAME = "RandomForest"

tr = pd.read_csv("data/processed/train_candidates_final.csv")
te = pd.read_csv("data/processed/student_test_candidates_final.csv")
print("train rows:", len(tr), "| test rows:", len(te), "| test trips:", te.trip_id.nunique())

fit = tr[tr.obs_has_true == 1]
model = get_models()[MODEL_NAME]
t0 = time.time()
model.fit(fit[FEATURES], fit["label"])
train_t = time.time() - t0

t0 = time.time()
proba = model.predict_proba(te[FEATURES])[:, 1]
infer_t = time.time() - t0
pred = (proba >= 0.5).astype(int)
y = te["label"]

print(f"top1      {top1_accuracy(te, proba):.4f}")
print(f"accuracy  {accuracy_score(y, pred):.4f}")
print(f"macro_f1  {f1_score(y, pred, average='macro'):.4f}")
print(f"precision {precision_score(y, pred, zero_division=0):.4f}")
print(f"recall    {recall_score(y, pred, zero_division=0):.4f}")
print(f"log_loss  {log_loss(y, proba, labels=[0, 1]):.4f}")
print(f"train {train_t:.0f}s | infer {infer_t:.2f}s")

print("\nper-trip top-1:")
for t in sorted(te.trip_id.unique()):
    m = (te.trip_id == t).values
    print(t, round(top1_accuracy(te[m], proba[m]), 3))