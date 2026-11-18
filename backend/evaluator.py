"""Server-side evaluation. The only place the reference route is used."""
import numpy as np

from .data_loader import Trip
from .geo import make_projector, point_to_segments
from .ml.pipeline import PredictionResult


def evaluate(res: PredictionResult, trip: Trip) -> dict:
    if trip.route is None or len(trip.route) == 0:
        return {"has_reference": False}
    arcs, nodes, route = trip.arcs, trip.nodes, trip.route
    pred = res.route_arcs

    # Per-point reference label ASSUMPTION: nearest arc on the reference route.
    proj = make_projector(trip.track[:, 0].mean(), trip.track[:, 1].mean())
    px, py = proj(trip.track[:, 0], trip.track[:, 1])
    ax, ay = proj(nodes[arcs[route, 0], 0], nodes[arcs[route, 0], 1])
    bx, by = proj(nodes[arcs[route, 1], 0], nodes[arcs[route, 1], 1])
    labels = np.array([route[np.argmin(point_to_segments(px[i], py[i], ax, ay, bx, by))]
                       for i in range(len(px))])
    top1 = float(np.mean(res.point_arcs == labels))

    ps, rs = set(pred), set(int(x) for x in route)
    tp = len(ps & rs)
    prec = tp / len(ps) if ps else 0.0
    rec = tp / len(rs) if rs else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    if len(pred) > 1:
        ok = sum(arcs[pred[i], 1] == arcs[pred[i + 1], 0] for i in range(len(pred) - 1))
        cont = float(ok / (len(pred) - 1))
    else:
        cont = None
    return {
        "has_reference": True,
        "point_top1_accuracy": top1,
        "route_arc_precision": prec, "route_arc_recall": rec, "route_arc_f1": f1,
        "route_continuity": cont,
        "n_pred_arcs": len(pred), "n_ref_arcs": len(rs),
    }
