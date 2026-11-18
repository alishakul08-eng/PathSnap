import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="PathSnap", layout="wide")
st.title("PathSnap: Route Reconstruction")

api_url = st.sidebar.text_input("API address", "http://localhost:8000")

try:
    health = requests.get(f"{api_url}/health", timeout=5).json()
    st.sidebar.success(f"API connected (mock={health.get('mock')})")
    trip_ids = requests.get(f"{api_url}/trips", timeout=5).json()["trip_ids"]
except Exception as e:
    st.sidebar.error("Cannot reach API")
    st.error(f"Could not connect to {api_url}: {e}")
    st.stop()

trip_id = st.selectbox("Select evaluation trip", trip_ids)

if st.button("Run prediction"):
    with st.spinner("Running pipeline..."):
        r = requests.post(f"{api_url}/predict", json={"trip_id": trip_id}, timeout=60)
    if r.status_code != 200:
        st.error(f"API error {r.status_code}: {r.json().get('detail')}")
    else:
        data = r.json()
        st.metric("Observations", data["n_observations"])
        df = pd.DataFrame(data["predictions"])
        st.subheader("Predictions")
        st.dataframe(df, use_container_width=True)
        st.subheader("Decoded route (arc IDs)")
        st.write(" → ".join(str(a) for a in data["route_arc_ids"]))
