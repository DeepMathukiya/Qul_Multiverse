"""Quality Analysis dashboard (frontend layer — display only).

Run:
    streamlit run frontend/streamlit_dashboard.py

Requires the backend (port 5000) and datascience (port 8100) services.
"""

from __future__ import annotations

import base64
import sys
import time
from pathlib import Path

import streamlit as st

# Allow `frontend.` imports when launched via `streamlit run`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from frontend.api_client import DEFAULT_BACKEND_URL, BackendClient  # noqa: E402
from frontend.result_views import render_full_result, show_image_bytes  # noqa: E402

st.set_page_config(
    page_title="QA Inspection Dashboard",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Computer-Vision Quality Analysis")

# ---------------- sidebar ----------------

with st.sidebar:
    st.header("Settings")
    backend_url = st.text_input("Backend URL", value=DEFAULT_BACKEND_URL)
    client = BackendClient(backend_url)

    if client.health():
        st.success("backend online")
    else:
        st.error("backend offline — start it with:\n`uvicorn backend.main:app --port 5000`")

    mode = st.radio("Acquisition mode", ["Live (two phones)", "Upload images"])

# ---------------- live mode ----------------

continuous = False
refresh_sec = 5

if mode == "Live (two phones)":
    st.subheader("Live stream preview")

    continuous = st.toggle(
        "🔁 Continuous streaming mode (auto-inspect + auto-refresh)",
        value=True,
    )
    refresh_sec = st.slider(
        "Dashboard refresh interval (s)", 2, 30, 5, disabled=not continuous
    )

    # Tell the backend loop to run (or stop) to match the toggle.
    try:
        stream_status = client.stream_status()
        if continuous and not stream_status.get("running"):
            stream_status = client.start_stream()
        elif not continuous and stream_status.get("running"):
            stream_status = client.stop_stream()
    except Exception:
        stream_status = {}

    if continuous:
        st.caption(
            f"backend loop: {'🟢 running' if stream_status.get('running') else '🔴 stopped'} · "
            f"inspections so far: {stream_status.get('inspection_count', 0)} · "
            f"{stream_status.get('last_error') or 'ok'}"
        )

    col_refresh, col_inspect = st.columns([1, 1])
    refresh = col_refresh.button("🔄 Refresh preview", use_container_width=True)
    inspect_clicked = col_inspect.button(
        "▶️ Run inspection now", type="primary", use_container_width=True
    )

    try:
        pair = client.latest_pair()
    except Exception as exc:
        pair = None
        st.error(f"could not fetch frames: {exc}")

    if pair is None:
        st.info(
            "Waiting for two devices to stream… point both phone apps at "
            f"`{backend_url}/upload`."
        )
    else:
        sync_icon = "🟢" if pair["synchronized"] else "🟠"
        st.caption(
            f"{sync_icon} pair Δt = {pair['time_delta_ms']} ms · "
            f"vertical: {pair['vertical_device_id']} · "
            f"horizontal: {pair['horizontal_device_id']}"
        )
        col_v, col_h = st.columns(2)
        show_image_bytes(
            col_v, base64.b64decode(pair["vertical_image_b64"]), "Vertical camera"
        )
        show_image_bytes(
            col_h, base64.b64decode(pair["horizontal_image_b64"]), "Horizontal camera"
        )

    if inspect_clicked:
        with st.spinner("Running full inspection pipeline…"):
            try:
                st.session_state["result"] = client.inspect_live()
            except Exception as exc:
                st.error(f"inspection failed: {exc}")

# ---------------- upload mode ----------------

else:
    st.subheader("Upload a stereo pair")
    col_v, col_h = st.columns(2)
    vertical_file = col_v.file_uploader("Vertical camera image", type=["jpg", "jpeg", "png"])
    horizontal_file = col_h.file_uploader("Horizontal camera image", type=["jpg", "jpeg", "png"])

    if st.button("▶️ Run inspection", type="primary",
                 disabled=not (vertical_file and horizontal_file)):
        with st.spinner("Running full inspection pipeline…"):
            try:
                st.session_state["result"] = client.inspect_upload(
                    vertical_file.getvalue(), horizontal_file.getvalue()
                )
            except Exception as exc:
                st.error(f"inspection failed: {exc}")

# ---------------- results ----------------

st.divider()

result = None
if continuous:
    # Continuous mode: always show the backend's newest loop result.
    try:
        result = client.latest_inspection()
    except Exception:
        result = None
if result is None:
    result = st.session_state.get("result")
if result is None:
    # Fall back to the most recent inspection stored on the backend.
    try:
        result = client.latest_inspection()
    except Exception:
        result = None

if result:
    st.caption(f"Inspection `{result['inspection_id']}` · {result['created_at']}")
    render_full_result(result)
else:
    st.info("No inspection yet — run one above.")

# Continuous mode: refresh the whole dashboard on a timer so the newest
# inspection and live previews keep flowing in.
if continuous:
    time.sleep(refresh_sec)
    st.rerun()
