"""
build_features.py - adds all model features to the labelled candidate CSVs.

Every feature is computed from the GPS .track and the road network (.nodes/.arcs) only.
Nothing here reads the .route file, label or obs_has_true, so the same code can run
on evaluator trips inside the API (call add_features(cands, track, nodes, arcs) per trip).

Usage:
  python build_features.py "data/raw/Student folder" train        data/processed/train_candidates_labeled.csv        data/processed/train_candidates_final.csv
  python build_features.py "data/raw/Student folder" student_test data/processed/student_test_candidates_labeled.csv data/processed/student_test_candidates_final.csv
"""
import glob, os, sys
import numpy as np
import pandas as pd

FEATURES = [
    "distance_m", "heading_difference", "candidate_rank",          # from Person 1
    "arc_len_m", "proj_frac", "speed_mps", "dt_prev", "heading_change_deg",
    "dist_prev", "dist_next", "dist_min_obs", "dist_ratio_best",
    "has_reverse_twin", "same_arc_as_prev_best", "connects_from_prev_best",
    "arc_persistence",
]
# n_candidates is NOT included: every observation has exactly 5 candidates (constant).

def _xy(lon, lat, lon0, lat0):
    return np.c_[(lon - lon0) * 111320.0 * np.cos(np.radians(lat0)), (lat - lat0) * 110540.0]

def _seg(P, A, B):
    """distance (m) and projection fraction of points P onto segments A-B (all Nx2)."""
    AB = B - A
    L2 = (AB ** 2).sum(1)
    t = np.where(L2 > 1e-9, ((P - A) * AB).sum(1) / np.maximum(L2, 1e-9), 0.0)
    t = np.clip(t, 0.0, 1.0)
    return np.linalg.norm(P - (A + t[:, None] * AB), axis=1), t

def add_features(c, trk, nodes, arcs):
    """c: candidate rows of ONE trip (cols trip_id, observation_id, candidate_rank,
    candidate_arc_id, distance_m, heading_difference, timestamp). Returns c with new columns."""
    c = c.copy()
    lon0, lat0 = trk.lon.mean(), trk.lat.mean()
    P = _xy(trk.lon.values, trk.lat.values, lon0, lat0)
    N = _xy(nodes.lon.values, nodes.lat.values, lon0, lat0)
    ts = trk.ts.values.astype(float)
    n = len(P)
    fn, tn = arcs.from_node.values, arcs.to_node.values
    obs = c.observation_id.values
    arc = c.candidate_arc_id.values
    A, B = N[fn[arc]], N[tn[arc]]

    d_own, frac = _seg(P[obs], A, B)
    c["arc_len_m"] = np.linalg.norm(B - A, axis=1)
    c["proj_frac"] = frac

    # point-level temporal features
    dt = np.r_[0.0, np.diff(ts)]
    step = np.r_[0.0, np.linalg.norm(np.diff(P, axis=0), axis=1)]
    speed = np.where(dt > 0, step / np.maximum(dt, 1e-9), 0.0)
    v_in = np.r_[[[0, 0]], np.diff(P, axis=0)]
    v_out = np.r_[np.diff(P, axis=0), [[0, 0]]]
    ang = np.degrees(np.abs(np.arctan2(v_in[:, 0], v_in[:, 1]) - np.arctan2(v_out[:, 0], v_out[:, 1])))
    ang = np.where(ang > 180, 360 - ang, ang)
    moving = (np.linalg.norm(v_in, axis=1) > 0.5) & (np.linalg.norm(v_out, axis=1) > 0.5)
    hchg = np.where(moving, ang, 0.0)
    c["dt_prev"] = dt[obs]
    c["speed_mps"] = speed[obs]
    c["heading_change_deg"] = hchg[obs]

    # distance of the previous / next GPS point to THE SAME arc (fallback: own distance at trip ends)
    prv, nxt = np.clip(obs - 1, 0, n - 1), np.clip(obs + 1, 0, n - 1)
    dprev, _ = _seg(P[prv], A, B)
    dnext, _ = _seg(P[nxt], A, B)
    c["dist_prev"] = np.where(obs == 0, c.distance_m.values, dprev)
    c["dist_next"] = np.where(obs == n - 1, c.distance_m.values, dnext)

    # relative to the other candidates of the same observation
    dmin = c.groupby("observation_id")["distance_m"].transform("min")
    c["dist_min_obs"] = c.distance_m - dmin
    c["dist_ratio_best"] = c.distance_m / (dmin + 1.0)

    # opposite-direction twin of this arc among this observation's candidates
    ends = set(zip(obs.tolist(), fn[arc].tolist(), tn[arc].tolist()))
    c["has_reverse_twin"] = [int((o, t, f) in ends) for o, f, t in zip(obs, fn[arc], tn[arc])]

    # connectivity to the previous observation's closest candidate (rank 1)
    best = c[c.candidate_rank == 1].drop_duplicates("observation_id").set_index("observation_id").candidate_arc_id
    pb = pd.Series(obs - 1).map(best).values            # NaN for obs 0
    has = ~np.isnan(pb)
    pb_i = np.where(has, pb, 0).astype(int)
    c["same_arc_as_prev_best"] = (has & (arc == pb_i)).astype(int)
    c["connects_from_prev_best"] = (has & (tn[pb_i] == fn[arc]) & (arc != pb_i)).astype(int)

    # how many neighbouring observations (+-5) also list this arc as a candidate
    M = 1_000_000
    key = obs.astype(np.int64) * M + arc.astype(np.int64)
    keyset = pd.Index(key)
    pers = np.zeros(len(c), dtype=int)
    for k in list(range(-5, 0)) + list(range(1, 6)):
        pers += keyset.isin(key + k * M)
    c["arc_persistence"] = pers
    c["_d_check"] = d_own
    return c

def read_trip(folder):
    d = os.path.basename(folder); p = f"{folder}/{d}"
    trk = pd.read_csv(p + ".track", sep=r"\s+", header=None, names=["lon", "lat", "ts"])
    nodes = pd.read_csv(p + ".nodes", sep=r"\s+", header=None, names=["lon", "lat"])
    arcs = pd.read_csv(p + ".arcs", sep=r"\s+", header=None, names=["from_node", "to_node"])
    return trk, nodes, arcs

def main(raw_root, split, cfile, out):
    c = pd.read_csv(cfile)
    parts = []
    for folder in sorted(glob.glob(f"{raw_root}/{split}/*")):
        t = int(os.path.basename(folder))
        trk, nodes, arcs = read_trip(folder)
        parts.append(add_features(c[c.trip_id == t], trk, nodes, arcs))
    f = pd.concat(parts).loc[c.index]                     # restore original row order
    diff = (f["_d_check"] - f["distance_m"]).abs()
    print(f"{split}: rows={len(f)} | max |recomputed distance - distance_m| = {diff.max():.3f} m (mean {diff.mean():.4f})")
    f = f.drop(columns="_d_check")
    f.to_csv(out, index=False)

if __name__ == "__main__":
    main(*sys.argv[1:5])
