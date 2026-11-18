import time
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, log_loss

FEATURES = [
    "distance_m", "heading_difference", "candidate_rank", "arc_len_m",
    "proj_frac", "speed_mps", "dt_prev", "heading_change_deg",
    "dist_prev", "dist_next", "dist_min_obs", "dist_ratio_best",
    "has_reverse_twin", "same_arc_as_prev_best",
    "connects_from_prev_best", "arc_persistence",
]

def top1_accuracy(va, proba):
    """Per observation: does the highest-probability candidate equal the true one?
    Only observations that actually have a true candidate are counted."""
    d = va[["trip_id", "observation_id", "label", "obs_has_true"]].copy()
    d["p"] = proba
    d = d[d.obs_has_true == 1]
    idx = d.groupby(["trip_id", "observation_id"])["p"].idxmax()
    return d.loc[idx, "label"].mean()

def cross_validate_model(name, model, df, n_splits=5):
    rows = []
    gkf = GroupKFold(n_splits)
    for fold, (tr_i, va_i) in enumerate(gkf.split(df, df["label"], df["trip_id"])):
        tr = df.iloc[tr_i]
        tr = tr[tr.obs_has_true == 1]          # train only where a true candidate exists
        va = df.iloc[va_i]                      # validate on every row
        t0 = time.time()
        model.fit(tr[FEATURES], tr["label"])
        train_t = time.time() - t0
        t0 = time.time()
        proba = model.predict_proba(va[FEATURES])[:, 1]
        infer_t = time.time() - t0
        pred = (proba >= 0.5).astype(int)
        y = va["label"]
        rows.append({
            "model": name, "fold": fold,
            "top1_acc": top1_accuracy(va, proba),
            "accuracy": accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, average="macro"),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "log_loss": log_loss(y, proba, labels=[0, 1]),
            "train_time_s": train_t, "infer_time_s": infer_t,
        })
        print(f"fold {fold}: top1={rows[-1]['top1_acc']:.3f} macro_f1={rows[-1]['macro_f1']:.3f} "
              f"train={train_t:.0f}s", flush=True)
    return pd.DataFrame(rows)
