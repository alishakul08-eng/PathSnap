"""
PathSnap ML pipeline  --  TEMPLATE.

Real:        candidate generation, feature extraction (basic), output contract.
PLACEHOLDER: score_candidates()  -> replace with your trained classifier's predict_proba
             decode()            -> replace with your HMM + Viterbi

The pipeline NEVER receives the reference route. Keep it that way.
"""
import time
from dataclasses import dataclass, field

import numpy as np

from ..geo import make_projector, point_to_segments


@dataclass
class PredictionResult:
    point_arcs: np.ndarray                 # (n,) predicted arc id per GPS point, -1 = none
    route_arcs: list[int]                  # de-duplicated ordered arc sequence
    timings_ms: dict = field(default_factory=dict)
    model_info: dict = field(default_factory=dict)


class PathSnapPipeline:
    CANDIDATE_RADIUS_M = 100.0
    MAX_CANDIDATES = 10

    def __init__(self, artifacts_dir=None):
        self.artifacts_dir = artifacts_dir
        self.model = None
        self.load_artifacts()

    # ---- TODO(ML): load trained model, scaler, HMM config from artifacts_dir ----
    def load_artifacts(self):
        # e.g. self.model = joblib.load(self.artifacts_dir / "model.joblib")
        pass

    # ------------------------------------------------------------------ public
    def predict(self, track: np.ndarray, nodes: np.ndarray, arcs: np.ndarray) -> PredictionResult:
        from . import v2_infer
        point_arcs, tm = v2_infer.predict_arrays(track, nodes, arcs)
        route = []
        for a in point_arcs:
            if a >= 0 and (not route or route[-1] != a):
                route.append(int(a))
        return PredictionResult(
            point_arcs=point_arcs, route_arcs=route, timings_ms=tm,
            model_info={"status": "TRAINED MODEL",
                        "classifier": "HistGradientBoosting v2 (21 features incl. direction features)",
                        "decoder": "Viterbi (path-aware, U-turn penalty)"},
        )

    # --------------------------------------------------------- pipeline stages
    def generate_candidates(self, px, py, seg) -> list[np.ndarray]:
        """For each GPS point: arc ids within CANDIDATE_RADIUS_M, nearest first."""
        ax, ay, bx, by = seg
        r = self.CANDIDATE_RADIUS_M
        keep = np.nonzero(
            (np.maximum(ax, bx) >= px.min() - r) & (np.minimum(ax, bx) <= px.max() + r) &
            (np.maximum(ay, by) >= py.min() - r) & (np.minimum(ay, by) <= py.max() + r))[0]
        out = []
        for i in range(len(px)):
            d = point_to_segments(px[i], py[i], ax[keep], ay[keep], bx[keep], by[keep])
            ok = np.nonzero(d <= r)[0]
            ok = ok[np.argsort(d[ok])][: self.MAX_CANDIDATES]
            out.append(keep[ok])
        return out

    def extract_features(self, px, py, seg, cands) -> list[np.ndarray]:
        """Per-candidate features [dist_m, arc_len_m, cos(heading, arc_dir)].
        TODO(ML): add speed, previous-point distance, bearing difference, etc."""
        ax, ay, bx, by = seg
        n = len(px)
        feats = []
        for i, ids in enumerate(cands):
            if len(ids) == 0:
                feats.append(np.zeros((0, 3))); continue
            d = point_to_segments(px[i], py[i], ax[ids], ay[ids], bx[ids], by[ids])
            dx, dy = bx[ids] - ax[ids], by[ids] - ay[ids]
            ln = np.hypot(dx, dy)
            hx, hy = px[min(i + 1, n - 1)] - px[max(i - 1, 0)], py[min(i + 1, n - 1)] - py[max(i - 1, 0)]
            hn = np.hypot(hx, hy)
            cos = (dx * hx + dy * hy) / (np.where(ln > 0, ln, 1) * hn) if hn > 0 else np.zeros(len(ids))
            feats.append(np.column_stack([d, ln, cos]))
        return feats

    def score_candidates(self, feats) -> list[np.ndarray]:
        """PLACEHOLDER. Replace with e.g. self.model.predict_proba(F)[:, 1] then normalise per point."""
        out = []
        for F in feats:
            if len(F) == 0:
                out.append(np.zeros(0)); continue
            z = -(F[:, 0] ** 2) / (2 * 15.0 ** 2)
            e = np.exp(z - z.max())
            out.append(e / e.sum())
        return out

    def decode(self, cands, scores, arcs) -> np.ndarray:
        """PLACEHOLDER: argmax per point. Replace with HMM + Viterbi.
        Hidden states = candidate arcs per point; emission = classifier prob;
        transition = road connectivity / network distance; handle empty candidate sets & gaps."""
        return np.array([c[np.argmax(s)] if len(c) else -1 for c, s in zip(cands, scores)], dtype=np.int64)
