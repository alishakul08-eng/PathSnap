import sys, warnings
from pathlib import Path
from types import SimpleNamespace
import joblib
import sklearn._loss._loss as _l
sys.modules['_loss'] = _l
warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import knn_pathsnap as kp
import decode_hgb as dh
import train_v2 as tv2

_B = None

def _bundle():
    global _B
    if _B is None:
        _B = joblib.load(HERE / "model_v2.joblib")
        kp.CFG = SimpleNamespace(prob_eps=_B["prob_eps"], hops=_B["hops"], features=_B["features"])
    return _B

def predict_trip(trip_dir):
    b = _bundle()
    trip_dir = Path(trip_dir)
    tid = int(trip_dir.name)
    kp.TRIP_DIRS[tid] = trip_dir
    df, _ = kp.build_trip(tid, "demo", trip_dir, b["radius"], b["max_candidates"])
    df = df.sort_values(["trip_id", "observation_id", "candidate_rank"]).reset_index(drop=True)
    df = tv2.add_direction_features(df, kp.TRIP_DIRS)
    p = kp.predict_prob(b["model"], df[b["features"]], b["pos_keep_frac"], b["neg_keep_frac"])
    ch = dh.decode_all(df, p, b["lam"], b["hops"], b["uturn"])
    out = df.loc[ch, ["trip_id", "observation_id", "timestamp", "candidate_arc_id"]].copy()
    out["p"] = p[ch]
    return out


import time
import tempfile
import numpy as np


def predict_arrays(track, nodes, arcs):
    """Same real pipeline as predict_trip, but takes the loaded arrays (no .route ever involved).
    Returns (one arc id per ORIGINAL gps point, -1 if dropped/none ; timings in ms)."""
    b = _bundle()
    tm = {}
    t_all = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        tid = 90000001
        d = Path(tmp) / str(tid)
        d.mkdir()
        np.savetxt(d / "t.track", track, fmt="%.17g")
        np.savetxt(d / "t.nodes", nodes, fmt="%.17g")
        np.savetxt(d / "t.arcs", arcs, fmt="%d")
        kp._nets.clear()
        kp.TRIP_DIRS.clear()
        kp.TRIP_DIRS[tid] = d

        t = time.perf_counter()
        df, _ = kp.build_trip(tid, "demo", d, b["radius"], b["max_candidates"])
        tm["candidates"] = (time.perf_counter() - t) * 1e3
        if len(df) == 0:
            return np.full(len(track), -1, dtype=np.int64), tm
        df = df.sort_values(["trip_id", "observation_id", "candidate_rank"]).reset_index(drop=True)

        t = time.perf_counter()
        df = tv2.add_direction_features(df, kp.TRIP_DIRS)
        tm["features"] = (time.perf_counter() - t) * 1e3

        t = time.perf_counter()
        p = kp.predict_prob(b["model"], df[b["features"]], b["pos_keep_frac"], b["neg_keep_frac"])
        tm["scoring"] = (time.perf_counter() - t) * 1e3

        t = time.perf_counter()
        ch = dh.decode_all(df, p, b["lam"], b["hops"], b["uturn"])
        tm["decode"] = (time.perf_counter() - t) * 1e3
        kp._nets.clear()
        kp.TRIP_DIRS.clear()

    lut = dict(zip(df.loc[ch, "timestamp"].to_numpy(), df.loc[ch, "candidate_arc_id"].to_numpy()))
    out = np.array([lut.get(ts, -1) for ts in track[:, 2]], dtype=np.int64)
    tm["total"] = (time.perf_counter() - t_all) * 1e3
    return out, tm
