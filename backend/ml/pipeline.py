"""
PathSnap ML pipeline  --  v2 HGB model + direction features + Viterbi decoding.

Drop-in replacement for backend/ml/pipeline.py (same PredictionResult / PathSnapPipeline API).
Self-contained: it does NOT import ..geo, so geometry is guaranteed identical to training.
The pipeline never receives the reference route.

Artifacts needed (one file):  <artifacts_dir>/model_v2.joblib      (from  hgb_v2/model_v2.joblib)
It carries the model, feature list, radius, max candidates, prior-correction fractions and the
decoder settings (lam, uturn, hops) chosen during training.
"""
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

R_EARTH = 6371008.8
DIR_WINDOWS = (3, 8, 20)


@dataclass
class PredictionResult:
    point_arcs: np.ndarray                 # (n,) predicted arc id per GPS point, -1 = none
    route_arcs: list[int]                  # de-duplicated ordered arc sequence
    timings_ms: dict = field(default_factory=dict)
    model_info: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------- geometry
def _make_projector(lon0, lat0):
    c = np.cos(np.radians(lat0))
    return lambda lon, lat: (np.radians(np.asarray(lon, float) - lon0) * R_EARTH * c,
                             np.radians(np.asarray(lat, float) - lat0) * R_EARTH)


def _dist_frac(px, py, ax, ay, bx, by):
    """Distance (m) and projection fraction (0..1) from point(s) to segment(s) A->B."""
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    safe = np.where(l2 > 0, l2, 1.0)
    t = np.clip(np.where(l2 > 0, ((px - ax) * dx + (py - ay) * dy) / safe, 0.0), 0.0, 1.0)
    return np.hypot(px - (ax + t * dx), py - (ay + t * dy)), t


class PathSnapPipeline:
    # hyper-parameters identical to training (overridden from the saved bundle)
    CANDIDATE_RADIUS_M = 100.0
    MAX_CANDIDATES = 40
    HEADING_HALF_WINDOW = 2
    MIN_DISPLACEMENT_M = 3.0
    PERSIST_WINDOW = 5

    def __init__(self, artifacts_dir=None):
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else Path(__file__).parent / "artifacts"
        self.model = None
        self.load_artifacts()

    # ------------------------------------------------------------------ artifacts
    def load_artifacts(self):
        b = joblib.load(self.artifacts_dir / "model_v2.joblib")
        self.model = b["model"]
        self.features = list(b["features"])
        self.pos_keep = float(b["pos_keep_frac"])
        self.neg_keep = float(b["neg_keep_frac"])
        self.prob_eps = float(b["prob_eps"])
        self.CANDIDATE_RADIUS_M = float(b["radius"])
        self.MAX_CANDIDATES = int(b["max_candidates"])
        self.lam = float(b.get("lam", 8.0))
        self.uturn = float(b.get("uturn", 10.0))
        self.hops = int(b.get("hops", 3))

    # ------------------------------------------------------------------ public
    def predict(self, track: np.ndarray, nodes: np.ndarray, arcs: np.ndarray) -> PredictionResult:
        tm = {}
        t0 = time.perf_counter()
        track = np.asarray(track, float)
        nodes = np.asarray(nodes, float)
        arcs = np.asarray(arcs).astype(np.int64)
        n_raw = len(track)
        point_arcs = np.full(n_raw, -1, dtype=np.int64)

        # --- clean exactly like training: drop invalid rows, sort by time, drop duplicate timestamps
        ok = np.isfinite(track).all(axis=1) & (np.abs(track[:, 0]) <= 180) & (np.abs(track[:, 1]) <= 90)
        idx = np.nonzero(ok)[0]
        idx = idx[np.argsort(track[idx, 2], kind="stable")]
        _, first = np.unique(track[idx, 2], return_index=True)
        idx = idx[np.sort(first)]                      # idx[k] = original row of cleaned point k
        if len(idx) == 0 or len(arcs) == 0:
            return PredictionResult(point_arcs, [], {"total": 0.0}, self._info())
        tr = track[idx]

        # --- projection of track and network
        proj = _make_projector(tr[:, 0].mean(), tr[:, 1].mean())
        px, py = proj(tr[:, 0], tr[:, 1])
        ax, ay = proj(nodes[arcs[:, 0], 0], nodes[arcs[:, 0], 1])
        bx, by = proj(nodes[arcs[:, 1], 0], nodes[arcs[:, 1], 1])
        tm["setup"] = (time.perf_counter() - t0) * 1e3

        t = time.perf_counter()
        obs, gid, X = self.build_features(px, py, tr[:, 2], (ax, ay, bx, by), arcs, len(nodes))
        tm["candidates_features"] = (time.perf_counter() - t) * 1e3

        if len(obs) == 0:
            tm["total"] = (time.perf_counter() - t0) * 1e3
            return PredictionResult(point_arcs, [], tm, self._info())

        t = time.perf_counter()
        p = self.score_candidates(X)
        tm["scoring"] = (time.perf_counter() - t) * 1e3

        t = time.perf_counter()
        obs_ids, chosen_arcs = self.decode(obs, gid, p, arcs)
        tm["decode"] = (time.perf_counter() - t) * 1e3

        point_arcs[idx[obs_ids]] = chosen_arcs
        tm["total"] = (time.perf_counter() - t0) * 1e3

        route = []
        for a in point_arcs:
            if a >= 0 and (not route or route[-1] != a):
                route.append(int(a))
        return PredictionResult(point_arcs=point_arcs, route_arcs=route, timings_ms=tm, model_info=self._info())

    def _info(self):
        return {"status": "TRAINED", "classifier": type(self.model.steps[-1][1]).__name__,
                "decoder": f"Viterbi (lam={self.lam:g}, uturn={self.uturn:g}, hops={self.hops})",
                "n_features": len(self.features)}

    # ------------------------------------------------- candidates + features (port of training)
    def build_features(self, px, py, t, seg, arcs, n_nodes):
        """Returns (obs_index[k], candidate_arc_id[k], DataFrame[k x features]) for all candidate rows,
        ordered by (observation, candidate rank). Identical logic to training."""
        ax, ay, bx, by = seg
        n = len(px)
        radius, kmax = self.CANDIDATE_RADIUS_M, self.MAX_CANDIDATES

        key = arcs[:, 0] * n_nodes + arcs[:, 1]
        twin_all = np.isin(arcs[:, 1] * n_nodes + arcs[:, 0], key).astype(int)

        keep = np.nonzero((np.maximum(ax, bx) >= px.min() - radius) & (np.minimum(ax, bx) <= px.max() + radius) &
                          (np.maximum(ay, by) >= py.min() - radius) & (np.minimum(ay, by) <= py.max() + radius))[0]
        kax, kay, kbx, kby = ax[keep], ay[keep], bx[keep], by[keep]
        adx, ady = kbx - kax, kby - kay
        alen = np.hypot(adx, ady)

        # direction of every arc (for direction features)
        fl = np.hypot(bx - ax, by - ay)
        fl = np.where(fl > 0, fl, 1.0)
        ux, uy = (bx - ax) / fl, (by - ay) / fl

        # travel heading + speed over a +-hwin window; forward-fill heading when nearly stationary
        hwin = self.HEADING_HALF_WINDOW
        i0, i1 = np.maximum(np.arange(n) - hwin, 0), np.minimum(np.arange(n) + hwin, n - 1)
        hx, hy = px[i1] - px[i0], py[i1] - py[i0]
        disp = np.hypot(hx, hy)
        dtw = t[i1] - t[i0]
        speed = np.where(dtw > 0, disp / np.where(dtw > 0, dtw, 1), 0.0)
        valid = disp >= self.MIN_DISPLACEMENT_M
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

        # pass 1: candidates per observation (nearest first, ties broken by heading)
        cand = []
        for i in range(n):
            dd, fr = _dist_frac(px[i], py[i], kax, kay, kbx, kby)
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
        best_gid = [int(keep[c[0][0]]) if c is not None else -1 for c in cand]

        # pass 2: features
        names = ["candidate_rank", "distance_m", "heading_difference", "arc_len_m", "proj_frac", "dt_prev",
                 "speed_mps", "heading_change_deg", "dist_prev", "dist_next", "dist_min_obs", "dist_ratio_best",
                 "has_reverse_twin", "same_arc_as_prev_best", "connects_from_prev_best", "arc_persistence",
                 "dir_cos3", "dir_cos8", "dir_cos20", "dir_along8", "dir_along20"]
        cols = {k: [] for k in names}
        obs_l, gid_l = [], []
        pw = self.PERSIST_WINDOW
        for i, c in enumerate(cand):
            if c is None:
                continue
            sel, dsel, fsel, hd = c
            gid = keep[sel]
            k = len(sel)
            ip, inx = max(i - 1, 0), min(i + 1, n - 1)
            dprev = _dist_frac(px[ip], py[ip], kax[sel], kay[sel], kbx[sel], kby[sel])[0]
            dnext = _dist_frac(px[inx], py[inx], kax[sel], kay[sel], kbx[sel], kby[sel])[0]
            bp = best_gid[i - 1] if i > 0 else -1
            same = (gid == bp).astype(int) if bp >= 0 else np.zeros(k, int)
            conn = (arcs[gid, 0] == arcs[bp, 1]).astype(int) if bp >= 0 else np.zeros(k, int)
            nb = [j for j in range(max(0, i - pw), min(n, i + pw + 1)) if j != i]
            persist = np.array([sum(g in gsets[j] for j in nb) for g in gid.tolist()])
            dmin = dsel.min()
            vals = {
                "candidate_rank": np.arange(1, k + 1), "distance_m": dsel, "heading_difference": hd,
                "arc_len_m": alen[sel], "proj_frac": fsel, "dt_prev": np.full(k, dt_prev[i]),
                "speed_mps": np.full(k, speed[i]), "heading_change_deg": np.full(k, heading_change[i]),
                "dist_prev": dprev, "dist_next": dnext, "dist_min_obs": dsel - dmin,
                "dist_ratio_best": dsel / (dmin + 1.0), "has_reverse_twin": twin_all[gid],
                "same_arc_as_prev_best": same, "connects_from_prev_best": conn, "arc_persistence": persist,
            }
            for w in DIR_WINDOWS:
                a0, a1 = max(i - w, 0), min(i + w, n - 1)
                ddx, ddy = px[a1] - px[a0], py[a1] - py[a0]
                along = ddx * ux[gid] + ddy * uy[gid]
                mag = np.hypot(ddx, ddy)
                vals[f"dir_cos{w}"] = along / mag if mag > 1.0 else np.zeros(k)
                if w in (8, 20):
                    vals[f"dir_along{w}"] = np.clip(along, -300, 300)
            for kk in names:
                cols[kk].append(vals[kk])
            obs_l.append(np.full(k, i))
            gid_l.append(gid)

        if not obs_l:
            return np.zeros(0, int), np.zeros(0, int), pd.DataFrame(columns=self.features)
        X = pd.DataFrame({k: np.concatenate(v) for k, v in cols.items()})[self.features]
        return np.concatenate(obs_l), np.concatenate(gid_l), X

    # ------------------------------------------------------------------ scoring
    def score_candidates(self, X: pd.DataFrame, batch=50_000) -> np.ndarray:
        """Model probability that each candidate is the true arc, corrected for the training
        under-sampling of negatives (same formula as in training)."""
        q = np.concatenate([self.model.predict_proba(X.iloc[s:s + batch])[:, 1]
                            for s in range(0, len(X), batch)])
        q = np.clip(q, self.prob_eps, 1 - self.prob_eps)
        p = q * self.neg_keep / (q * self.neg_keep + (1 - q) * self.pos_keep)
        return np.clip(p, self.prob_eps, 1 - self.prob_eps)

    # ------------------------------------------------------------------ decoding
    def decode(self, obs, gid, p, arcs):
        """Viterbi over observations that have candidates. Emission = log p; transition cost:
        same arc / directly connected -> 0, reachable in 2..hops -> lam/2, otherwise lam,
        opposite-direction twin (U-turn) -> uturn.  Returns (observation ids, chosen arc ids)."""
        order = np.argsort(arcs[:, 0], kind="stable")
        sfrom, sto = arcs[order, 0], arcs[order, 1]
        cache = {}

        def reach(a):
            if a not in cache:
                end = int(arcs[a, 1])
                d, frontier, seen = {end: 1}, [end], {end}
                for h in range(2, self.hops + 1):
                    nxt = []
                    for nd in frontier:
                        for x in sto[np.searchsorted(sfrom, nd, "left"):np.searchsorted(sfrom, nd, "right")]:
                            x = int(x)
                            if x not in seen:
                                seen.add(x)
                                d[x] = h
                                nxt.append(x)
                    frontier = nxt
                cache[a] = d
            return cache[a]

        lp_all = np.log(np.clip(p, 1e-6, 1.0))
        starts = np.flatnonzero(np.r_[True, obs[1:] != obs[:-1]])
        ends = np.r_[starts[1:], len(obs)]
        gids = [gid[s:e] for s, e in zip(starts, ends)]
        lps = [lp_all[s:e] for s, e in zip(starts, ends)]
        obs_ids = obs[starts]

        lam, ut = self.lam, self.uturn
        m = len(gids)
        score = lps[0].copy()
        back = [None]
        for i in range(1, m):
            ga, gb = gids[i - 1], gids[i]
            sb = arcs[gb, 0]
            C = np.empty((len(ga), len(gb)))
            for ia, a in enumerate(ga):
                d = reach(int(a))
                hs = np.fromiter((d.get(int(s), 0) for s in sb), dtype=np.int64, count=len(sb))
                row = np.where(hs == 1, 0.0, np.where(hs > 1, lam / 2.0, lam))
                rev = (arcs[gb, 0] == arcs[a, 1]) & (arcs[gb, 1] == arcs[a, 0])
                row[rev] = ut
                row[gb == a] = 0.0
                C[ia] = row
            tot = score[:, None] - C
            arg = tot.argmax(axis=0)
            score = tot[arg, np.arange(len(gb))] + lps[i]
            back.append(arg)
        path = np.empty(m, dtype=np.int64)
        path[-1] = int(np.argmax(score))
        for i in range(m - 1, 0, -1):
            path[i - 1] = back[i][path[i]]
        chosen = np.array([gids[i][path[i]] for i in range(m)], dtype=np.int64)
        return obs_ids, chosen