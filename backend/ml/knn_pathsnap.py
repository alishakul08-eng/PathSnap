"""
PathSnap - KNN candidate classifier, built DIRECTLY from the raw trip folders (no CSV needed).

Put this file in the folder that contains  train/  and  student_test/  and run:
        python knn_pathsnap.py
(or:    python knn_pathsnap.py --root "C:\\Users\\ABHISHEK KINAGI\\Downloads\\Student folder\\Student folder")

Each trip folder must hold  <name>.track  <name>.nodes  <name>.arcs  <name>.route
Pipeline:  raw files -> candidate arcs + features (in memory) -> KNN -> CV + student-test evaluation

Install once:   pip install numpy pandas scikit-learn joblib

Selection metrics : Observation-level Top-1 Accuracy, Macro F1, Route Continuity
Supporting        : Log Loss, Precision, Recall, Candidate Miss Rate, Training / Inference time
The reference route (.route) is used ONLY to create `label` / `obs_has_true` and for scoring.
Evaluator-test data is never read (only the splits named in --splits).
"""
import argparse
import json
import pickle
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, log_loss, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
warnings.filterwarnings("ignore")
R_EARTH = 6371008.8
CFG = None                  # filled in main()
TRIP_DIRS = {}              # trip_id -> folder (used to load .arcs lazily for route continuity)

ALL_FEATURES = [
    "candidate_rank", "distance_m", "heading_difference", "arc_len_m", "proj_frac",
    "dt_prev", "speed_mps", "heading_change_deg", "dist_prev", "dist_next",
    "dist_min_obs", "dist_ratio_best", "has_reverse_twin",
    "same_arc_as_prev_best", "connects_from_prev_best", "arc_persistence",
]
TEMPORAL = ["dt_prev", "heading_change_deg", "dist_prev", "dist_next",
            "same_arc_as_prev_best", "connects_from_prev_best", "arc_persistence"]


# =============================================================================== geometry
def make_projector(lon0, lat0):
    c = np.cos(np.radians(lat0))
    return lambda lon, lat: (np.radians(np.asarray(lon, float) - lon0) * R_EARTH * c,
                             np.radians(np.asarray(lat, float) - lat0) * R_EARTH)


def dist_frac(px, py, ax, ay, bx, by):
    """Distance (m) and projection fraction (0..1) from point(s) to segment(s) A->B."""
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    safe = np.where(l2 > 0, l2, 1.0)
    t = np.clip(np.where(l2 > 0, ((px - ax) * dx + (py - ay) * dy) / safe, 0.0), 0.0, 1.0)
    return np.hypot(px - (ax + t * dx), py - (ay + t * dy)), t


# =============================================================================== loading
def find_file(d, ext):
    m = sorted(Path(d).glob(f"*{ext}"))
    return m[0] if m else None


def load_trip(d):
    ft, fn, fa, fr = (find_file(d, e) for e in (".track", ".nodes", ".arcs", ".route"))
    if not (ft and fn and fa):
        raise FileNotFoundError(f"missing .track/.nodes/.arcs in {d}")
    track = np.loadtxt(ft, ndmin=2)
    nodes = np.loadtxt(fn, ndmin=2)
    arcs = np.loadtxt(fa, dtype=np.int64, ndmin=2)
    route = np.loadtxt(fr, dtype=np.int64, ndmin=1) if fr else None
    return track, nodes, arcs, route


def clean_track(track):
    """Drop invalid rows, sort by time, drop duplicate timestamps (use the SAME cleaning in the API)."""
    n0 = len(track)
    ok = np.isfinite(track).all(axis=1) & (np.abs(track[:, 0]) <= 180) & (np.abs(track[:, 1]) <= 90)
    track = track[ok]
    track = track[np.argsort(track[:, 2], kind="stable")]
    _, first = np.unique(track[:, 2], return_index=True)
    track = track[np.sort(first)]
    return track, n0 - len(track)


# =============================================================================== labels
def monotone_reference_arcs(px, py, rax, ray, rbx, rby, chunk=400):
    """Per-point reference arc = route arc minimising summed distance, with the route index
    non-decreasing over time. Returns indices into the route array."""
    n, R = len(px), len(rax)
    D = np.empty((n, R))
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        D[s:e] = dist_frac(px[s:e, None], py[s:e, None], rax[None], ray[None], rbx[None], rby[None])[0]
    idx = np.arange(R)
    cost = D[0].copy()
    back = np.zeros((n, R), dtype=np.int32)
    for i in range(1, n):
        pm = np.minimum.accumulate(cost)
        back[i] = np.maximum.accumulate(np.where(cost == pm, idx, 0))
        cost = D[i] + pm
    j = int(np.argmin(cost))
    out = np.empty(n, dtype=np.int64)
    for i in range(n - 1, -1, -1):
        out[i] = j
        j = back[i, j]
    return out


# =============================================================================== candidates + features
def build_trip(trip_id, split, d, radius, kmax, hwin=2, min_disp=3.0, persist_w=5):
    track, nodes, arcs, route = load_trip(d)
    track, n_dropped = clean_track(track)
    n = len(track)
    lon, lat, t = track[:, 0], track[:, 1], track[:, 2]
    proj = make_projector(lon.mean(), lat.mean())
    px, py = proj(lon, lat)
    ax, ay = proj(nodes[arcs[:, 0], 0], nodes[arcs[:, 0], 1])
    bx, by = proj(nodes[arcs[:, 1], 0], nodes[arcs[:, 1], 1])

    # reverse-direction twin flag per arc (network property, no labels involved)
    N = len(nodes)
    key = arcs[:, 0].astype(np.int64) * N + arcs[:, 1]
    twin_all = np.isin(arcs[:, 1].astype(np.int64) * N + arcs[:, 0], key).astype(int)

    keep = np.nonzero((np.maximum(ax, bx) >= px.min() - radius) & (np.minimum(ax, bx) <= px.max() + radius) &
                      (np.maximum(ay, by) >= py.min() - radius) & (np.minimum(ay, by) <= py.max() + radius))[0]
    kax, kay, kbx, kby = ax[keep], ay[keep], bx[keep], by[keep]
    adx, ady = kbx - kax, kby - kay
    alen = np.hypot(adx, ady)

    # travel heading + speed over a +-hwin window; forward-fill heading when nearly stationary
    i0, i1 = np.maximum(np.arange(n) - hwin, 0), np.minimum(np.arange(n) + hwin, n - 1)
    hx, hy = px[i1] - px[i0], py[i1] - py[i0]
    disp = np.hypot(hx, hy)
    dtw = t[i1] - t[i0]
    speed = np.where(dtw > 0, disp / np.where(dtw > 0, dtw, 1), 0.0)
    valid = disp >= min_disp
    hxn, hyn = np.zeros(n), np.zeros(n)
    hxn[valid], hyn[valid] = hx[valid] / disp[valid], hy[valid] / disp[valid]
    last = None
    for i in range(n):
        if valid[i]:
            last = i
        elif last is not None:
            hxn[i], hyn[i], valid[i] = hxn[last], hyn[last], True
    if valid.any() and not valid.all():
        first = int(np.argmax(valid))
        hxn[:first], hyn[:first] = hxn[first], hyn[first]
    hvec_ok = np.hypot(hxn, hyn) > 0
    heading_change = np.zeros(n)
    for i in range(1, n):
        if hvec_ok[i] and hvec_ok[i - 1]:
            heading_change[i] = np.degrees(np.arccos(np.clip(hxn[i] * hxn[i - 1] + hyn[i] * hyn[i - 1], -1, 1)))
    dt_prev = np.empty(n)
    dt_prev[1:] = np.diff(t)
    dt_prev[0] = dt_prev[1] if n > 1 else 0.0

    # reference arc per point (labels only)
    ref_arc = None
    if route is not None and len(route) > 0:
        ref_arc = route[monotone_reference_arcs(px, py, ax[route], ay[route], bx[route], by[route])]

    # ---- pass 1: candidates per observation (nearest first, ties broken by heading) ----
    cand = []
    for i in range(n):
        dd, fr = dist_frac(px[i], py[i], kax, kay, kbx, kby)
        sel = np.nonzero(dd <= radius)[0]
        if len(sel) == 0:
            cand.append(None)
            continue
        ln = alen[sel]
        cosv = np.where(ln > 0, (adx[sel] * hxn[i] + ady[sel] * hyn[i]) / np.where(ln > 0, ln, 1), 0.0)
        hd = np.degrees(np.arccos(np.clip(cosv, -1, 1)))
        order = np.lexsort((hd, dd[sel]))[:kmax]
        sel, hd = sel[order], hd[order]
        cand.append((sel, dd[sel], fr[sel], hd))
    gsets = [set(keep[c[0]].tolist()) if c is not None else set() for c in cand]
    best_gid = [int(keep[c[0][0]]) if c is not None else -1 for c in cand]   # nearest candidate (label-free)

    # ---- pass 2: features ----
    cols = {k: [] for k in ["trip_id", "split", "observation_id", "timestamp", "candidate_arc_id"] + ALL_FEATURES +
            ["label", "obs_has_true"]}
    n_zero, n_with_true = 0, 0
    for i, c in enumerate(cand):
        if c is None:
            n_zero += 1
            continue
        sel, dsel, fsel, hd = c
        gid = keep[sel]
        k = len(sel)
        ip, inx = max(i - 1, 0), min(i + 1, n - 1)
        dprev = dist_frac(px[ip], py[ip], kax[sel], kay[sel], kbx[sel], kby[sel])[0]
        dnext = dist_frac(px[inx], py[inx], kax[sel], kay[sel], kbx[sel], kby[sel])[0]
        bp = best_gid[i - 1] if i > 0 else -1
        same = (gid == bp).astype(int) if bp >= 0 else np.zeros(k, int)
        conn = (arcs[gid, 0] == arcs[bp, 1]).astype(int) if bp >= 0 else np.zeros(k, int)
        nb = [j for j in range(max(0, i - persist_w), min(n, i + persist_w + 1)) if j != i]
        persist = np.array([sum(g in gsets[j] for j in nb) for g in gid.tolist()])
        has_true = int(ref_arc is not None and np.any(gid == ref_arc[i]))
        n_with_true += has_true
        dmin = dsel.min()
        vals = {
            "trip_id": np.full(k, trip_id), "split": np.full(k, split), "observation_id": np.full(k, i),
            "timestamp": np.full(k, t[i]), "candidate_arc_id": gid,
            "candidate_rank": np.arange(1, k + 1), "distance_m": dsel, "heading_difference": hd,
            "arc_len_m": alen[sel], "proj_frac": fsel, "dt_prev": np.full(k, dt_prev[i]),
            "speed_mps": np.full(k, speed[i]), "heading_change_deg": np.full(k, heading_change[i]),
            "dist_prev": dprev, "dist_next": dnext, "dist_min_obs": dsel - dmin,
            "dist_ratio_best": dsel / (dmin + 1.0), "has_reverse_twin": twin_all[gid],
            "same_arc_as_prev_best": same, "connects_from_prev_best": conn, "arc_persistence": persist,
            "label": (gid == ref_arc[i]).astype(int) if ref_arc is not None else np.full(k, -1),
            "obs_has_true": np.full(k, has_true if ref_arc is not None else -1),
        }
        for kk, v in vals.items():
            cols[kk].append(v)

    df = pd.DataFrame({k: np.concatenate(v) for k, v in cols.items()}) if cols["trip_id"] else pd.DataFrame()
    summ = {
        "trip_id": trip_id, "split": split, "n_points_raw": n + n_dropped, "n_points_used": n,
        "n_points_dropped": n_dropped, "n_nodes": N, "n_arcs": len(arcs),
        "n_route_arcs": 0 if route is None else len(route),
        "duration_s": float(t[-1] - t[0]), "median_dt": float(np.median(np.diff(t))) if n > 1 else np.nan,
        "max_dt": float(np.max(np.diff(t))) if n > 1 else np.nan,
        "n_candidate_rows": len(df), "obs_no_candidates": n_zero, "obs_with_true": n_with_true,
        "candidate_miss_rate": (1 - n_with_true / n) if ref_arc is not None else np.nan,
    }
    return df, summ


def load_split(root, split, radius, kmax, cache_dir, use_cache):
    sdir = Path(root) / split
    if not sdir.is_dir():
        raise SystemExit(f"[error] folder not found: {sdir}\n  --root must contain '{split}/'")
    trips = sorted(p for p in sdir.iterdir() if p.is_dir())
    dfs, sums, t0 = [], [], time.time()
    for k, d in enumerate(trips, 1):
        tid = int(d.name)
        TRIP_DIRS[tid] = d
        cf = Path(cache_dir) / f"{split}_{d.name}_r{int(radius)}_k{kmax}.pkl"
        if use_cache and cf.exists():
            df, summ = pickle.load(open(cf, "rb"))
        else:
            if find_file(d, ".route") is None:
                print(f"  [skip] {split}/{d.name}: no .route file (cannot label)")
                continue
            try:
                df, summ = build_trip(tid, split, d, radius, kmax)
            except Exception as e:
                print(f"  [skip] {split}/{d.name}: {e}")
                continue
            if use_cache:
                cf.parent.mkdir(parents=True, exist_ok=True)
                pickle.dump((df, summ), open(cf, "wb"))
        if len(df):
            dfs.append(df)
        sums.append(summ)
        print(f"  [{split} {k}/{len(trips)}] trip {d.name}: points={summ['n_points_used']} "
              f"rows={summ['n_candidate_rows']} miss_rate={summ['candidate_miss_rate']:.3f} ({time.time() - t0:.0f}s)")
    out = pd.concat(dfs, ignore_index=True)
    out = out.sort_values(["trip_id", "observation_id", "candidate_rank"]).reset_index(drop=True)
    return out, pd.DataFrame(sums)


# =============================================================================== route continuity
class Network:
    def __init__(self, arcs):
        self.arcs = arcs
        o = np.argsort(arcs[:, 0], kind="stable")
        self._from, self._to = arcs[o, 0], arcs[o, 1]

    def succ(self, node):
        return self._to[np.searchsorted(self._from, node, "left"):np.searchsorted(self._from, node, "right")]

    def hops(self, a, b, max_hops):
        """1 = b starts exactly where a ends; h = reachable via h arcs; 0 = not within max_hops."""
        target = self.arcs[b, 0]
        frontier = {int(self.arcs[a, 1])}
        if target in frontier:
            return 1
        seen = set(frontier)
        for h in range(2, max_hops + 1):
            nxt = set()
            for nd in frontier:
                nxt.update(int(x) for x in self.succ(nd))
            frontier = nxt - seen
            if target in frontier:
                return h
            if not frontier:
                break
            seen |= frontier
        return 0


_nets = {}


def get_network(trip_id):
    if trip_id not in _nets:
        _nets[trip_id] = Network(np.loadtxt(find_file(TRIP_DIRS[int(trip_id)], ".arcs"), dtype=np.int64, ndmin=2))
    return _nets[trip_id]


def continuity(seq_by_trip):
    strict = relaxed = pairs = 0
    for trip, seq in seq_by_trip.items():
        seq = [a for i, a in enumerate(seq) if i == 0 or a != seq[i - 1]]
        if len(seq) < 2:
            continue
        net = get_network(trip)
        for a, b in zip(seq[:-1], seq[1:]):
            h = net.hops(a, b, CFG.hops)
            strict += (h == 1)
            relaxed += (h > 0)
            pairs += 1
    return (strict / pairs, relaxed / pairs) if pairs else (np.nan, np.nan)


# =============================================================================== evaluation
def evaluate(df, p):
    """Top-1 accuracy (strict: obs whose true arc is not a candidate count as wrong), Macro F1
    (binary wrong/true, predicted class 1 = top-1 candidate per observation), route continuity, etc."""
    df = df.reset_index(drop=True).assign(p=np.asarray(p))
    top_idx = df.groupby(["trip_id", "observation_id"])["p"].idxmax().to_numpy()
    top = df.loc[top_idx]
    y_true = df["label"].to_numpy()
    y_top = np.zeros(len(df), dtype=int)
    y_top[top_idx] = 1
    y_thr = (df["p"].to_numpy() >= 0.5).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_top, labels=[0, 1], zero_division=0)
    _, _, f1_thr, _ = precision_recall_fscore_support(y_true, y_thr, labels=[0, 1], zero_division=0)

    pred_seq = top.groupby("trip_id")["candidate_arc_id"].apply(list).to_dict()
    ref_seq = (df[df.label == 1].sort_values(["trip_id", "observation_id"])
               .groupby("trip_id")["candidate_arc_id"].apply(list).to_dict())
    c_strict, c_relax = continuity(pred_seq)
    r_strict, r_relax = continuity(ref_seq)
    return {
        "n_obs": int(len(top)), "n_rows": int(len(df)),
        "top1_acc": float((top.label == 1).mean()),
        "macro_f1": float(f1.mean()),
        "route_continuity": c_strict, "route_continuity_khop": c_relax,
        "top1_acc_given_true_present": float(top.loc[top.obs_has_true == 1, "label"].mean()),
        "macro_precision": float(pr.mean()), "macro_recall": float(rc.mean()),
        "precision_pos": float(pr[1]), "recall_pos": float(rc[1]),
        "macro_f1_thr0.5": float(f1_thr.mean()),
        "log_loss": float(log_loss(y_true, df["p"].clip(1e-6, 1 - 1e-6), labels=[0, 1])),
        "cand_miss_rate": float(1 - top["obs_has_true"].mean()),
        "reference_continuity": r_strict, "reference_continuity_khop": r_relax,
        "confusion_matrix_top1": confusion_matrix(y_true, y_top, labels=[0, 1]).tolist(),
    }, top


# =============================================================================== model
def make_classifier():
    name = CFG.model_name
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=CFG.k, weights="distance", p=2,
                                    algorithm="brute", n_jobs=-1)
    if name == "logreg":
        return LogisticRegression(C=1.0, max_iter=2000, random_state=CFG.seed)
    if name == "rf":
        return RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                      n_jobs=-1, random_state=CFG.seed)
    if name == "hgb":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                              random_state=CFG.seed)
    if name == "mlp":
        return MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4,
                             learning_rate_init=1e-3, batch_size=512,
                             early_stopping=False, max_iter=60,
                             random_state=CFG.seed)
    raise ValueError(name)


def make_model():
    prep = ColumnTransformer([("log", FunctionTransformer(np.log1p, validate=False), CFG.log_cols)],
                             remainder="passthrough")
    return Pipeline([("prep", prep), ("scale", StandardScaler()),
                     ("f32", FunctionTransformer(np.float32, validate=False)),
                     ("clf", make_classifier())])


def subsample_rows(y, max_rows, seed):
    """Sample both classes while ensuring the training set has both classes."""
    rng = np.random.default_rng(seed)

    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)

    if len(pos) == 0 or len(neg) == 0:
        raise ValueError(
            f"Training data has only one class: "
            f"positive={len(pos)}, negative={len(neg)}"
        )

    if max_rows < 2:
        raise ValueError("max_train_rows must be at least 2")

    # Keep both classes, allocating remaining capacity to negatives.
    n_pos = min(len(pos), max(1, max_rows // 2))
    n_neg = min(len(neg), max_rows - n_pos)

    # If negatives are insufficient, use remaining capacity for positives.
    if n_neg < max_rows - n_pos:
        n_pos = min(len(pos), max_rows - n_neg)

    selected_pos = rng.choice(pos, n_pos, replace=False)
    selected_neg = rng.choice(neg, n_neg, replace=False)

    idx = np.sort(np.concatenate([selected_pos, selected_neg]))

    # Sampling probabilities for prior correction.
    pos_keep = n_pos / len(pos)
    neg_keep = n_neg / len(neg)

    return idx, pos_keep, neg_keep


def predict_prob(model, X, pos_keep, neg_keep, batch=50_000):
    if list(model.classes_) != [0, 1]:
        raise ValueError(
            f"KNN requires both classes [0, 1]. "
            f"Found: {model.classes_}"
        )

    ps = [
        model.predict_proba(X.iloc[s:s + batch])[:, 1]
        for s in range(0, len(X), batch)
    ]

    q = np.clip(np.concatenate(ps), CFG.prob_eps, 1 - CFG.prob_eps)

    # Correct probabilities for class-dependent sampling.
    p = (
        q * neg_keep /
        (q * neg_keep + (1 - q) * pos_keep)
    )

    return np.clip(p, CFG.prob_eps, 1 - CFG.prob_eps)


def fit_model(tr_df):
    idx, pos_keep, neg_keep = subsample_rows(
        tr_df["label"].to_numpy(),
        CFG.max_train_rows,
        CFG.seed
    )

    sub = tr_df.iloc[idx]

    model = make_model()

    t0 = time.perf_counter()
    model.fit(sub[CFG.features], sub["label"])
    fit_s = time.perf_counter() - t0

    print(
        f"    training classes={model.classes_}, "
        f"positive={int((sub.label == 1).sum())}, "
        f"negative={int((sub.label == 0).sum())}"
    )

    return model, pos_keep, neg_keep, len(sub), fit_s


# =============================================================================== main
def main():
    global CFG
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent),
                    help="folder containing train/ and student_test/")
    ap.add_argument("--out-dir", default=None, help="default: <root>/knn_results")
    ap.add_argument("--radius", type=float, default=100.0, help="candidate radius (m)")
    ap.add_argument("--max-candidates", type=int, default=40)
    ap.add_argument("--feature-set", choices=["A", "B"], default="B",
                    help="B = all 16 features, A = per-point features only (drops temporal/context ones)")
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-train-rows", type=int, default=150_000)
    ap.add_argument("--cv-max-val-trips", type=int, default=8, help="0 = score all validation trips")
    ap.add_argument("--hops", type=int, default=3, help="k-hop tolerance for relaxed continuity")
    ap.add_argument("--prob-eps", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-cache", action="store_true", help="rebuild candidate tables instead of reusing the cache")
    ap.add_argument("--model", choices=["knn", "logreg", "rf", "hgb", "mlp"], default="knn")
    ap.add_argument("--export-csv", action="store_true", help="also write the built tables to CSV (optional)")
    ap.add_argument("--exclude-trips", type=int, nargs="*", default=[],help="training trip ids to drop (invalid reference route)")
    a = ap.parse_args()

    root = Path(a.root)
    out = Path(a.out_dir) if a.out_dir else root / "knn_results"
    out.mkdir(parents=True, exist_ok=True)
    feats = [c for c in ALL_FEATURES if a.feature_set == "B" or c not in TEMPORAL]
    log_cols = [c for c in ["distance_m", "arc_len_m", "dist_prev", "dist_next", "dist_min_obs",
                            "dist_ratio_best", "speed_mps", "dt_prev"] if c in feats]
    CFG = SimpleNamespace(k=a.k, seed=a.seed, hops=a.hops, prob_eps=a.prob_eps,
                      max_train_rows=a.max_train_rows, features=feats, log_cols=log_cols,
                      model_name=a.model)

    print(f"root: {root}")
    cache = out / "cache"
    print("\n[1/4] building candidates + features from the raw folders (cached after the first run)")
    train_df, train_sum = load_split(root, "train", a.radius, a.max_candidates, cache, not a.no_cache)
    student_df, student_sum = load_split(root, "student_test", a.radius, a.max_candidates, cache, not a.no_cache)
    assert not (set(train_df.trip_id) & set(student_df.trip_id)), "trip leakage between splits!"
    pd.concat([train_sum, student_sum]).to_csv(out / "trip_summary.csv", index=False)
    train_df = train_df[~train_df["trip_id"].isin(a.exclude_trips)].reset_index(drop=True)
    if a.export_csv:
        train_df.to_csv(out / "train_candidates.csv", index=False)
        student_df.to_csv(out / "student_candidates.csv", index=False)
    print(f"\ntrain  : {len(train_df):,} rows, {train_df.trip_id.nunique()} trips")
    print(f"student: {len(student_df):,} rows, {student_df.trip_id.nunique()} trips")
    print(f"features ({len(feats)}): {feats}")

    print("\n[2/4] cross-validation on training trips (GroupKFold by trip)")
    rng = np.random.default_rng(a.seed)
    cv_rows = []
    for fold, (tr, va) in enumerate(GroupKFold(n_splits=a.folds).split(train_df, groups=train_df["trip_id"]), 1):
        tr_df, va_df = train_df.iloc[tr], train_df.iloc[va]
        va_trips = va_df["trip_id"].unique()
        if a.cv_max_val_trips and len(va_trips) > a.cv_max_val_trips:
            va_trips = rng.choice(va_trips, a.cv_max_val_trips, replace=False)
            va_df = va_df[va_df["trip_id"].isin(va_trips)]
        model, pos_keep, neg_keep, n_fit, fit_s = fit_model(tr_df)
        t0 = time.perf_counter()
        p = predict_prob(model, va_df[feats], pos_keep, neg_keep)
        inf_s = time.perf_counter() - t0
        m, _ = evaluate(va_df, p)
        m.update(fold=fold, fit_s=fit_s, infer_s=inf_s, infer_ms_per_obs=1000 * inf_s / m["n_obs"],
                 n_fit_rows=n_fit, pos_keep_frac=pos_keep, neg_keep_frac=neg_keep,
                 n_val_trips=len(va_trips))
        cv_rows.append(m)
        print(f"  fold {fold}: top1={m['top1_acc']:.4f} macroF1={m['macro_f1']:.4f} "
              f"continuity={m['route_continuity']:.4f} logloss={m['log_loss']:.4f} "
              f"fit={fit_s:.0f}s infer={inf_s:.0f}s")
    cv_df = pd.DataFrame(cv_rows)
    cv_cols = ["top1_acc", "macro_f1", "route_continuity", "route_continuity_khop", "log_loss", "cand_miss_rate"]
    cv_summary = cv_df[cv_cols].agg(["mean", "std"]).T

    print("\n[3/4] final fit on all training trips -> student-test evaluation")
    model, pos_keep, neg_keep, n_fit, fit_s = fit_model(train_df)
    t0 = time.perf_counter()
    p_st = predict_prob(model, student_df[feats], pos_keep, neg_keep)
    inf_s = time.perf_counter() - t0
    sm, st_top = evaluate(student_df, p_st)
    sm.update(fit_s=fit_s, infer_s=inf_s, infer_ms_per_obs=1000 * inf_s / sm["n_obs"],
              n_fit_rows=n_fit, pos_keep_frac=pos_keep, neg_keep_frac=neg_keep)

    print("\n=== STUDENT-TEST : selection metrics ===")
    for k in ["top1_acc", "macro_f1", "route_continuity"]:
        print(f"  {k:30s}{sm[k]:.4f}")
    print("=== supporting ===")
    for k in ["route_continuity_khop", "reference_continuity", "log_loss", "macro_precision", "macro_recall",
              "precision_pos", "recall_pos", "cand_miss_rate", "top1_acc_given_true_present",
              "macro_f1_thr0.5", "fit_s", "infer_s", "infer_ms_per_obs"]:
        print(f"  {k:30s}{sm[k]:.4f}")
    print("  confusion matrix (rows=true 0/1, cols=pred 0/1):", sm["confusion_matrix_top1"])
    print("\n=== CV vs student-test ===")
    cmp = pd.DataFrame({"CV mean": cv_summary["mean"], "CV std": cv_summary["std"],
                        "student-test": pd.Series({k: sm[k] for k in cv_cols})})
    print(cmp.round(4).to_string())

    print("\n[4/4] saving")
    joblib.dump({"model": model, "features": feats, "log_cols": log_cols,
                 "pos_keep_frac": pos_keep, "neg_keep_frac": neg_keep,
                 "prob_eps": a.prob_eps, "radius": a.radius, "max_candidates": a.max_candidates},
                out / "knn_model.joblib")
    cv_df.to_csv(out / "knn_cv_folds.csv", index=False)
    st_top[["trip_id", "observation_id", "candidate_arc_id", "p", "label", "obs_has_true"]] \
        .to_csv(out / "knn_student_top1_predictions.csv", index=False)
    with open(out / "knn_results.json", "w") as fh:
        json.dump({"args": vars(a), "features": feats, "cv_mean": cv_summary["mean"].to_dict(),
                   "cv_std": cv_summary["std"].to_dict(), "student_test": sm}, fh, indent=2, default=float)
    print(f"saved to {out}")


if __name__ == "__main__":
    main()