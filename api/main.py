from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.schemas import PredictRequest, PredictResponse
from api import inference
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("pathsnap")

app = FastAPI(title="PathSnap API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

EVAL_TRIP_IDS = ["00000015", "00000017", "00000023", "00000030", "00000049", "00000059", "00000067", "00000073", "00000087", "00000094"]

@app.on_event("startup")
def startup():
    inference.load_model()
    log.info("Model loaded (mock=%s)", inference.USE_MOCK)

@app.get("/health")
def health():
    return {"status": "ok", "mock": inference.USE_MOCK}

@app.get("/trips")
def trips():
    return {"trip_ids": EVAL_TRIP_IDS}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if req.trip_id not in EVAL_TRIP_IDS:
        raise HTTPException(status_code=404, detail="Unknown trip_id")
    log.info("Predict request: trip %s", req.trip_id)
    try:
        return inference.run_pipeline(req.trip_id)
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        log.exception("Pipeline error for trip %s", req.trip_id)
        raise HTTPException(status_code=500, detail=str(e))