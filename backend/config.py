from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "trips"      # flat layout: <trip_id>.track/.nodes/.arcs/.route
ARTIFACTS_DIR = ROOT / "artifacts"      # trained model, scaler, HMM config go here later
HOST = "0.0.0.0"                        # reachable from Laptop 2
PORT = 8000
