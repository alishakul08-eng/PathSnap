"""
Path-aware (Viterbi) decoding on top of a saved PathSnap model. NO retraining.

Idea: instead of taking each GPS point's best arc independently, pick the whole arc
sequence of a trip that maximises   sum(log p_model)  -  sum(transition penalties).
Transition penalty between consecutive points (arc a -> arc b):
    same arc, or b starts where a ends (connected)  -> 0
    reachable within 2..hops arcs                    -> lam / 2
    not reachable                                    -> lam
lam is chosen on the STUDENT-TEST split (never on the evaluator data), then applied
once to the evaluator data.

Keep this file next to knn_pathsnap.py and run (from the Student folder):

  python decode_hgb.py --model-file "hgb_results\\knn_model.joblib" ^
     --root "C:\\Users\\ABHISHEK KINAGI\\Downloads\\evaluator_test" --split evaluator_test ^
     --tune-root "." --tune-split student_test
"""
import argparse
import json
import warnings
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd

import knn_pathsnap as kp

warnings.filterwarnings("ignore")


# ------------------------------------------------------------------ Viterbi
def make_reach(net, hops):
    cache = {}

    def reach(a):
        """dict: start-node -> h, meaning an arc starting at that node is reachable from arc a in h arcs."""
        if a not in cache:
            end = int(net.arcs[a, 1])
            d, frontier, seen = {end: 1}, [end], {end}
            for h in range(2, hops + 1):
                nxt = []
                for nd in frontier:
                    for x in net.succ(nd):
                        x = int(x)
                        if x not in seen:
                            seen.add(x)
                            d[x] = h
                            nxt.append(x)
                frontier = nxt
            cache[a] = d
        return cache[a]

    return reach


def decode_trip(net, gids, lps, lam, hops, uturn=0.0):
    """gids/lps: lists (one entry per observation) of candidate arc ids / log-probabilities.
    Returns the chosen candidate index (into each list entry) per observation."""
    reach = make_reach(net, hops)
    n = len(gids)
    score = lps[0].copy()
    back = [None]
    for i in range(1, n):
        ga, gb = gids[i - 1], gids[i]
        sb = net.arcs[gb, 0]
        C = np.empty((len(ga), len(gb)))
        for ia, a in enumerate(ga):
            d = reach(int(a))
            hs = np.fromiter((d.get(int(s), 0) for s in sb), dtype=np.int64, count=len(sb))
            row = np.where(hs == 1, 0.0, np.where(hs > 1, lam / 2.0, lam))
            # opposite-direction twin of arc a (U-turn): connected, but penalised separately
            rev = (net.arcs[gb, 0] == net.arcs[a, 1]) & (net.arcs[gb, 1] == net.arcs[a, 0])
            row[rev] = uturn
            row[gb == a] = 0.0
            C[ia] = row
        tot = score[:, None] - C
        arg = tot.argmax(axis=0)
        score = tot[arg, np.arange(len(gb))] + lps[i]
        back.append(arg)
    path = np.empty(n, dtype=np.int64)
    path[-1] = int(np.argmax(score))
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i][path[i]]
    return path


def decode_all(df, p, lam, hops, uturn=0.0):
    """Returns the row positions (index into df) of the chosen candidate for every observation."""
    lp_all = np.log(np.clip(p, 1e-6, 1.0))
    chosen = []
    for tid, sub in df.groupby("trip_id", sort=True):
        net = kp.get_network(tid)
        pos = sub.index.to_numpy()
        obs = sub["observation_id"].to_numpy()
        gid = sub["candidate_arc_id"].to_numpy()
        starts = np.flatnonzero(np.r_[True, obs[1:] != obs[:-1]])
        ends = np.r_[starts[1:], len(sub)]
        gids = [gid[s:e] for s, e in zip(starts, ends)]
        lps = [lp_all[pos[s:e]] for s, e in zip(starts, ends)]
        path = decode_trip(net, gids, lps, lam, hops, uturn)
        chosen.extend(pos[s] + k for s, k in zip(starts, path))
    return np.array(chosen, dtype=np.int64)


def score_decoded(df, p, chosen):
    """Reuse kp.evaluate: boost the chosen rows so they become each observation's top-1."""
    p2 = p.copy()
    p2[chosen] += 1.0
    return kp.evaluate(df, p2)


# ------------------------------------------------------------------ data
def build_split_from_folders(root, split, radius, kmax):
    sdir = Path(root) / split
    if not sdir.is_dir():
        raise SystemExit(f"[error] folder not found: {sdir}")
    dfs = []
    for d in sorted(p for p in sdir.iterdir() if p.is_dir()):
        tid = int(d.name)
        kp.TRIP_DIRS[tid] = d
        try:
            df, _ = kp.build_trip(tid, split, d, radius, kmax)
        except Exception as e:
            print(f"  [skip] {d.name}: {e}")
            continue
        if len(df):
            dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    return df.sort_values(["trip_id", "observation_id", "candidate_rank"]).reset_index(drop=True)


def show(tag, m):
    print(f"  {tag:22s} top1={m['top1_acc']:.4f}  macroF1={m['macro_f1']:.4f}  "
          f"continuity={m['route_continuity']:.4f}  continuity_khop={m['route_continuity_khop']:.4f}")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-file", required=True)
    ap.add_argument("--root", required=True, help="folder containing the evaluator split folder")
    ap.add_argument("--split", default="evaluator_test")
    ap.add_argument("--tune-root", default=".", help="folder containing student_test/ (used to choose lam)")
    ap.add_argument("--tune-split", default="student_test")
    ap.add_argument("--lams", type=float, nargs="*", default=[0.5, 1, 2, 3, 5, 8, 12])
    ap.add_argument("--hops", type=int, default=3)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()

    bundle = joblib.load(a.model_file)
    model, feats = bundle["model"], bundle["features"]
    pk, nk = bundle["pos_keep_frac"], bundle["neg_keep_frac"]
    radius, kmax = bundle["radius"], bundle["max_candidates"]
    kp.CFG = SimpleNamespace(prob_eps=bundle["prob_eps"], hops=a.hops, features=feats)
    print(f"model: {type(model.named_steps['clf']).__name__}")

    # ---------- 1) choose lam on student_test ----------
    print(f"\n[1/2] choosing lam on {a.tune_split} (not on the evaluator data)")
    cache = Path(a.model_file).resolve().parent / "cache"
    tdf, _ = kp.load_split(a.tune_root, a.tune_split, radius, kmax, cache, True)
    tp = kp.predict_prob(model, tdf[feats], pk, nk)
    base, _ = kp.evaluate(tdf, tp)
    show("baseline (argmax)", base)
    best = None
    for lam in a.lams:
        ch = decode_all(tdf, tp, lam, a.hops)
        m, _ = score_decoded(tdf, tp, ch)
        show(f"viterbi lam={lam:g}", m)
        key = (m["top1_acc"], m["route_continuity"])
        if best is None or key > best[0]:
            best = (key, lam)
    lam = best[1]
    print(f"  -> chosen lam = {lam:g}")

    # ---------- 2) apply to evaluator ----------
    print(f"\n[2/2] applying to {a.split}")
    kp._nets.clear()
    kp.TRIP_DIRS.clear()
    edf = build_split_from_folders(a.root, a.split, radius, kmax)
    ep = kp.predict_prob(model, edf[feats], pk, nk)
    if (edf["label"] == -1).all():
        print("  no .route files -> writing decoded predictions only")
    base_e, _ = kp.evaluate(edf, ep) if not (edf["label"] == -1).all() else (None, None)
    ch = decode_all(edf, ep, lam, a.hops)

    out = Path(a.out_dir) if a.out_dir else Path(a.root) / "eval_results"
    out.mkdir(parents=True, exist_ok=True)
    pred = edf.loc[ch, ["trip_id", "observation_id", "timestamp", "candidate_arc_id"]].copy()
    pred["p"] = ep[ch]
    pred.to_csv(out / "eval_viterbi_predictions.csv", index=False)

    if base_e is None:
        print(f"saved to {out}")
        return
    dec_e, top = score_decoded(edf, ep, ch)

    print("\n=== EVALUATOR-TEST : before vs after decoding ===")
    print(f"  {'metric':26s}{'argmax':>10s}{'viterbi':>10s}")
    for k in ["top1_acc", "macro_f1", "route_continuity", "route_continuity_khop",
              "macro_precision", "macro_recall"]:
        print(f"  {k:26s}{base_e[k]:10.4f}{dec_e[k]:10.4f}")
    print(f"  (log loss {base_e['log_loss']:.4f} is unchanged: decoding does not alter probabilities)")

    rows = []
    for tid, g in top.groupby("trip_id"):
        b = edf.loc[edf.trip_id == tid].copy()
        b["p"] = ep[b.index]
        bt = b.loc[b.groupby("observation_id")["p"].idxmax()]
        rows.append({"trip_id": tid, "n_obs": len(g),
                     "top1_argmax": float((bt.label == 1).mean()),
                     "top1_viterbi": float((g.label == 1).mean())})
    per_trip = pd.DataFrame(rows)
    print("\nPer-trip top-1:\n" + per_trip.round(4).to_string(index=False))
    per_trip.to_csv(out / "eval_viterbi_per_trip.csv", index=False)

    with open(out / "eval_viterbi_results.json", "w") as fh:
        json.dump({"lam": lam, "hops": a.hops, "argmax": base_e, "viterbi": dec_e},
                  fh, indent=2, default=float)
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
