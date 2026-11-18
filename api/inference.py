import os
import numpy as np
import pandas as pd

USE_MOCK = os.getenv("USE_MOCK", "1") == "1"
FEATURES: list = []

_model = None

def load_model():
    global _model
    if USE_MOCK:
        return
    import joblib
    _model = joblib.load("models/saved_models/final_model.joblib")

def get_candidates(trip_id: str) -> pd.DataFrame:
    # TODO Person 1: connect the real candidate/feature function
    if USE_MOCK:
        rng = np.random.default_rng(0)
        rows = [{"trip_id": trip_id, "observation_id": o, "candidate_arc_id": 100 + o * 3 + c,
                 "mock_feature": rng.random()} for o in range(5) for c in range(3)]
        return pd.DataFrame(rows)
    raise NotImplementedError("Connect Person 1's candidate/feature function")

def score_candidates(df: pd.DataFrame) -> pd.DataFrame:
    if USE_MOCK:
        df["predicted_probability"] = np.random.default_rng(1).random(len(df))
    else:
        df["predicted_probability"] = _model.predict_proba(df[FEATURES])[:, 1]
    return df

def decode_route(df: pd.DataFrame) -> list:
    # TODO Person 3: replace with HMM/Viterbi decoder
    best = df.loc[df.groupby("observation_id")["predicted_probability"].idxmax()]
    return best.sort_values("observation_id")["candidate_arc_id"].tolist()

def run_pipeline(trip_id: str) -> dict:
    df = score_candidates(get_candidates(trip_id))
    best = df.loc[df.groupby("observation_id")["predicted_probability"].idxmax()].sort_values("observation_id")
    route = decode_route(df)
    preds = [{"observation_id": int(r.observation_id), "predicted_arc_id": int(route[i]),
              "classifier_arc_id": int(r.candidate_arc_id), "probability": float(r.predicted_probability)}
             for i, r in enumerate(best.itertuples())]
    return {"trip_id": trip_id, "n_observations": len(preds), "predictions": preds,
            "route_arc_ids": [int(a) for a in route]}
