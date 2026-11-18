# ML PATHSNAP: DISTRIBUTED MAP-MATCHING & ROUTE RECONSTRUCTION 
## part of ML hackathon 

## Team Members

|No.| Name                     |   SRN         |
|:-:|--------------------------|---------------|
| 1 | ABHISHEK HANAMANT KINAGI | PES1UG24CS018 |
| 2 | ACHINTHYA M B            | PES1UG24CS019 |
| 3 | ALISHA KULSHRESTHA       | PES1UG24CS049 |
| 4 | ALLURU SHIVA SAMHITHA    | PES1UG24CS050 |

# PathSnap – API + UI scaffold

Two-laptop setup: **Laptop 1** = FastAPI (model + decoder + ground truth), **Laptop 2** = static UI.
The ML parts are placeholders (`backend/ml/pipeline.py`), clearly flagged in the UI.

## Install (both laptops, Python 3.10+)
    python -m venv venv
    venv\Scripts\activate          # Windows   (Linux/Mac: source venv/bin/activate)
    pip install -r requirements.txt

## Laptop 1 – run the API (from the project root)
    uvicorn backend.app:app --host 0.0.0.0 --port 8000
Docs: http://localhost:8000/docs   |   Request logs print in this terminal (show these to evaluators).
Find its IP with `ipconfig` (Windows) and allow inbound TCP 8000 in Windows Firewall.

## Laptop 2 – run the UI (only needs the `frontend/` folder, no pip install)
    python -m http.server 8080 --directory frontend
Open http://localhost:8080, put `http://<LAPTOP1_IP>:8000` in the API box, click Check connection.
(Needs internet for the Leaflet library and map tiles.)

## Trip data
Put trips in `data/trips/` as `<id>.track`, `.nodes`, `.arcs`, `.route` (flat). Any trip with
track+nodes+arcs appears in the UI automatically. `.route` is used only inside `evaluator.py`.

## Plugging in the ML (edit `backend/ml/pipeline.py`)
- `load_artifacts()` – load model/scaler/HMM config from `artifacts/`
- `extract_features()` – add your engineered features
- `score_candidates()` – replace with `model.predict_proba`
- `decode()` – replace with HMM + Viterbi
