"""
PathSnap v2: HGB + multi-window DIRECTION features + Viterbi decoding with U-turn penalty.

Why: most remaining errors are opposite-direction twin arcs (same road, wrong direction).
The old heading feature (+-2 points) is too noisy. Here, for every candidate arc and several
window sizes w, we measure how far the vehicle moved ALONG that arc's direction between
point i-w and point i+w. Twin arcs get opposite signs, so direction becomes easy to learn.

Keep next to knn_pathsnap.py and decode_hgb.py. Run from the Student folder:

  python train_v2.py --root "." ^
     --eval-root "C:\\Users\\ABHISHEK KINAGI\\Downloads\\evaluator_test" --eval-split evaluator_test

The base features are re-used from hgb_results\\cache (fast). Model/decoding settings are chosen
on student_test ONLY; the evaluator data is scored once at the end.
"""
import argparse
import json
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd

import knn_pathsnap as kp
import decode_hgb as dh

warnings.filterwarnings("ignore")

WINDOWS = (3, 8, 20)
NEW_FEATS = ["dir_cos3", "dir_cos8", "dir_cos20", "dir_along8", "dir_along20"]


def add_direction_features(df, trip_dirs):
    """Adds NEW_FEATS to a candidate table (needs the raw trip folders for the track + network)."""
    out = {c: np.zeros(len(df)) for c in NEW_FEATS}
    for tid, pos in df.groupby("trip_id").indices.items():
        track, nodes, arcs, _ = kp.load_trip(trip_dirs[int(tid)])
        track, _ = kp.clean_track(track)
        proj = kp.make_projector(track[:, 0].mean(), track[:, 1].mean())
        px, py = proj(track[:, 0], track[:, 1])
        ax, ay = proj(nodes[arcs[:, 0], 0], nodes[arcs[:, 0], 1])
        bx, by = proj(nodes[arcs[:, 1], 0], nodes[arcs[:, 1], 1])
        ln = np.hypot(bx - ax, by - ay)
        ln = np.where(ln > 0, ln, 1.0)
        ux, uy = (bx - ax) / ln, (by - ay) / ln

        sub = df.iloc[pos]
        gid = sub["candidate_arc_id"].to_numpy()
        obs = sub["observation_id"].to_numpy()
        n = len(px)
        for w in WINDOWS:
            i0, i1 = np.maximum(obs - w, 0), np.minimum(obs + w, n - 1)
            dx, dy = px[i1] - px[i0], py[i1] - py[i0]
            along = dx * ux[gid] + dy * uy[gid]
            mag = np.hypot(dx, dy)
            cos = np.where(mag > 1.0, along / np.where(mag > 0, mag, 1.0), 0.0)
            out[f"dir_cos{w}"][pos] = cos
            if w in (8, 20):
                out[f"dir_along{w}"][pos] = np.clip(along, -300, 300)
    df = df.copy()
    for c in NEW_FEATS:
        df[c] = out[c]
    return df


def show(tag, m):
    print(f"  {tag:34s} top1={m['top1_acc']:.4f}  macroF1={m['macro_f1']:.4f}  "
          f"continuity={m['route_continuity']:.4f}  khop={m['route_continuity_khop']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="folder containing train/ and student_test/")
    ap.add_argument("--eval-root", default=None)
    ap.add_argument("--eval-split", default="evaluator_test")
    ap.add_argument("--out-dir", default="hgb_v2")
    ap.add_argument("--cache-dir", default="hgb_results/cache")
    ap.add_argument("--max-train-rows", type=int, default=300_000)
    ap.add_argument("--model", default="hgb")
    ap.add_argument("--lams", type=float, nargs="*", default=[3, 5, 8])
    ap.add_argument("--uturns", type=float, nargs="*", default=[0, 2, 5, 10, 20])
    ap.add_argument("--hops", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--radius", type=float, default=100.0)
    ap.add_argument("--max-candidates", type=int, default=40)
    ap.add_argument("--exclude-trips", type=int, nargs="*", default=[])
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    feats = list(kp.ALL_FEATURES) + NEW_FEATS
    log_cols = [c for c in ["distance_m", "arc_len_m", "dist_prev", "dist_next", "dist_min_obs",
                            "dist_ratio_best", "speed_mps", "dt_prev"] if c in feats]
    kp.CFG = SimpleNamespace(k=25, seed=a.seed, hops=a.hops, prob_eps=1e-3, max_train_rows=a.max_train_rows,
                             features=feats, log_cols=log_cols, model_name=a.model)

    print("[1/4] loading cached base features + adding direction features")
    train_df, _ = kp.load_split(a.root, "train", a.radius, a.max_candidates, a.cache_dir, True)
    train_df = add_direction_features(train_df, kp.TRIP_DIRS)
    train_df = train_df[~train_df["trip_id"].isin(a.exclude_trips)].reset_index(drop=True)
    st_df, _ = kp.load_split(a.root, "student_test", a.radius, a.max_candidates, a.cache_dir, True)
    st_df = add_direction_features(st_df, kp.TRIP_DIRS)
    print(f"  train {len(train_df):,} rows / {train_df.trip_id.nunique()} trips;  "
          f"student {len(st_df):,} rows / {st_df.trip_id.nunique()} trips;  features={len(feats)}")

    print("\n[2/4] training")
    model, pk, nk, n_fit, fit_s = kp.fit_model(train_df)
    print(f"  fit on {n_fit:,} rows in {fit_s:.0f}s")

    print("\n[3/4] student_test: argmax vs decoding (settings chosen HERE)")
    sp = kp.predict_prob(model, st_df[feats], pk, nk)
    base, _ = kp.evaluate(st_df, sp)
    show("argmax", base)
    best = None
    for lam in a.lams:
        for ut in a.uturns:
            ch = dh.decode_all(st_df, sp, lam, a.hops, ut)
            m, _ = dh.score_decoded(st_df, sp, ch)
            show(f"viterbi lam={lam:g} uturn={ut:g}", m)
            key = (m["top1_acc"], m["route_continuity"])
            if best is None or key > best[0]:
                best = (key, lam, ut, m)
    _, lam, ut, _ = best
    print(f"  -> chosen lam={lam:g}, uturn={ut:g}")

    joblib.dump({"model": model, "features": feats, "log_cols": log_cols, "pos_keep_frac": pk,
                 "neg_keep_frac": nk, "prob_eps": 1e-3, "radius": a.radius,
                 "max_candidates": a.max_candidates, "lam": lam, "uturn": ut, "hops": a.hops},
                out / "model_v2.joblib")

    if not a.eval_root:
        print(f"\nsaved to {out}. (give --eval-root to score the evaluator data)")
        return

    print(f"\n[4/4] evaluator data ({a.eval_split}) - scored once")
    kp._nets.clear()
    kp.TRIP_DIRS.clear()
    ev = dh.build_split_from_folders(a.eval_root, a.eval_split, a.radius, a.max_candidates)
    ev = add_direction_features(ev, kp.TRIP_DIRS)
    ep = kp.predict_prob(model, ev[feats], pk, nk)
    ch = dh.decode_all(ev, ep, lam, a.hops, ut)
    pred = ev.loc[ch, ["trip_id", "observation_id", "timestamp", "candidate_arc_id"]].copy()
    pred["p"] = ep[ch]
    pred.to_csv(out / "eval_v2_predictions.csv", index=False)

    if (ev["label"] == -1).all():
        print("  no .route files -> predictions written, no metrics")
        return
    base_e, _ = kp.evaluate(ev, ep)
    dec_e, top = dh.score_decoded(ev, ep, ch)
    print("\n=== EVALUATOR-TEST (v2) ===")
    print(f"  {'metric':26s}{'argmax':>10s}{'viterbi':>10s}")
    for k in ["top1_acc", "macro_f1", "route_continuity", "route_continuity_khop",
              "macro_precision", "macro_recall"]:
        print(f"  {k:26s}{base_e[k]:10.4f}{dec_e[k]:10.4f}")
    print(f"  log_loss (unchanged by decoding) {base_e['log_loss']:.4f}")
    print("\n  previous result (v1, decoded): top1=0.8797  macroF1=0.9366  continuity=0.9324")

    rows = []
    for tid, g in top.groupby("trip_id"):
        rows.append({"trip_id": tid, "n_obs": len(g), "top1_viterbi": float((g.label == 1).mean())})
    per_trip = pd.DataFrame(rows)
    print("\nPer-trip top-1:\n" + per_trip.round(4).to_string(index=False))
    per_trip.to_csv(out / "eval_v2_per_trip.csv", index=False)
    with open(out / "eval_v2_results.json", "w") as fh:
        json.dump({"lam": lam, "uturn": ut, "argmax": base_e, "viterbi": dec_e}, fh, indent=2, default=float)
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
