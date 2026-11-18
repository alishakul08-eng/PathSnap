import os, json, time
from pathlib import Path
import numpy as np
import pandas as pd

USE_MOCK = os.getenv("USE_MOCK", "1") == "1"
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/saved_models/final_model.joblib"))
FEATURE_ORDER_PATH = Path(os.getenv("FEATURE_ORDER_PATH", "models/saved_models/feature_order.json"))
DATA_DIR = Path(os.getenv("DATA_DIR", "data/evaluator"))   # raw trip folders, never committed

FORBIDDEN = {"trip_id", "observation_id", "candidate_arc_id", "timestamp", "label", "obs_has_true"}
_model = None
FEATURES: list = []


def load_model():
    global _model, FEATURES
    if USE_MOCK:
        return
    import joblib
    for p in (MODEL_PATH, FEATURE_ORDER_PATH):
        if not p.exists():
            raise RuntimeError(f"Missing {p}. Get it from Person 2, or run with USE_MOCK=1.")
    _model = joblib.load(MODEL_PATH)
    FEATURES = json.loads(FEATURE_ORDER_PATH.read_text())
    bad = FORBIDDEN & set(FEATURES)
    if bad:
        raise RuntimeError(f"Forbidden columns in feature order: {bad}")


def get_candidates(trip_id: str) -> pd.DataFrame:
    if USE_MOCK:
        rng = np.random.default_rng(0)
        rows = [{"trip_id": trip_id, "observation_id": o, "candidate_arc_id": 100 + o * 3 + c,
                 "mock_feature": rng.random()} for o in range(5) for c in range(3)]
        return pd.DataFrame(rows)
    trip_dir = DATA_DIR / trip_id
    if not trip_dir.exists():
        raise FileNotFoundError(f"No raw files for trip {trip_id}")
    # TODO Person 1: cands = build_candidates(trip_id)   (.track, .nodes, .arcs only, never .route)
    # TODO: load track, nodes, arcs from trip_dir, then:
    # from preprocessing.build_features import add_features
    # return add_features(cands, track, nodes, arcs)
    raise NotImplementedError("Connect Person 1's build_candidates and add_features")


def score_candidates(df: pd.DataFrame) -> pd.DataFrame:
    if USE_MOCK:
        df["predicted_probability"] = np.random.default_rng(1).random(len(df))
        return df
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    df["predicted_probability"] = _model.predict_proba(df[FEATURES])[:, 1]
    return df


def decode_route(df: pd.DataFrame) -> list:
    # TODO Person 3: replace with the HMM/Viterbi decoder (needs the trip's .arcs graph)
    best = df.loc[df.groupby("observation_id")["predicted_probability"].idxmax()]
    return best.sort_values("observation_id")["candidate_arc_id"].tolist()


def run_pipeline(trip_id: str) -> dict:
    t0 = time.time()
    df = score_candidates(get_candidates(trip_id))
    best = df.loc[df.groupby("observation_id")["predicted_probability"].idxmax()].sort_values("observation_id")
    route = decode_route(df)
    if len(route) != len(best):
        raise ValueError("Decoder returned a route of the wrong length")
    preds = [{"observation_id": int(r.observation_id), "predicted_arc_id": int(route[i]),
              "classifier_arc_id": int(r.candidate_arc_id), "probability": float(r.predicted_probability)}
             for i, r in enumerate(best.itertuples())]
    return {"trip_id": trip_id, "mock": USE_MOCK, "n_observations": len(preds),
            "latency_s": round(time.time() - t0, 3), "predictions": preds,
            "route_arc_ids": [int(a) for a in route]}