import logging
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import config
from .data_loader import TripStore
from .evaluator import evaluate
from .ml.pipeline import PathSnapPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pathsnap")

app = FastAPI(title="PathSnap API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

store = TripStore(config.DATA_DIR)
pipeline = PathSnapPipeline(config.ARTIFACTS_DIR)


@app.middleware("http")
async def request_log(request: Request, call_next):
    t = time.perf_counter()
    resp = await call_next(request)
    log.info("REQ %s %s from=%s status=%s %.0fms", request.method, request.url.path,
             request.client.host if request.client else "?", resp.status_code,
             (time.perf_counter() - t) * 1e3)
    return resp


class PredictRequest(BaseModel):
    trip_id: str


@app.get("/health")
def health():
    return {"status": "ok", "n_trips": len(store.ids())}


@app.get("/trips")
def trips():
    out = []
    for tid in store.ids():
        tr = store.load(tid)
        out.append({"trip_id": tid, "n_points": int(len(tr.track))})   # no labels exposed
    return {"trips": out}


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        trip = store.load(req.trip_id)
    except KeyError:
        raise HTTPException(404, f"unknown trip_id {req.trip_id}")

    # The pipeline gets raw track + road network ONLY (no reference route).
    res = pipeline.predict(trip.track, trip.nodes, trip.arcs)
    metrics = evaluate(res, trip)             # reference route used here, server-side only

    geometry = [[[float(trip.nodes[trip.arcs[a, 0], 1]), float(trip.nodes[trip.arcs[a, 0], 0])],
                 [float(trip.nodes[trip.arcs[a, 1], 1]), float(trip.nodes[trip.arcs[a, 1], 0])]]
                for a in res.route_arcs]
    log.info("PREDICT trip=%s points=%d pred_arcs=%d metrics=%s",
             req.trip_id, len(trip.track), len(res.route_arcs), metrics)
    return {
        "trip_id": req.trip_id,
        "gps": [[float(r[1]), float(r[0])] for r in trip.track],      # [lat, lon]
        "predicted_route_arcs": res.route_arcs,
        "predicted_route_geometry": geometry,
        "metrics": metrics,
        "timings_ms": res.timings_ms,
        "model_info": res.model_info,
    }
