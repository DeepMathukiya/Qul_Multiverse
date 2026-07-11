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

    ocr_on = st.toggle(
        "🔤 OCR (Sarvam) enabled",
        value=True,
        help="Off = skip the slow OCR API call; QR decoding still runs.",
    )

# ---------------- live mode: two processed streams ----------------

continuous = False
refresh_sec = 1.0

if mode == "Live (two phones)":
    continuous = st.toggle(
        "🔁 Continuous streaming mode (auto-inspect + auto-refresh)",
        value=True,
    )
    refresh_fps = st.slider(
        "Stream rate (FPS)", 0.2, 5.0, 1.0, 0.2,
        disabled=not continuous,
        help="Inspections per second (drives both the backend loop and the "
             "dashboard refresh).",
    )
    refresh_sec = 1.0 / refresh_fps

    # The FPS slider drives the backend inspection cadence directly.
    try:
        if continuous:
            stream_status = client.start_stream(
                interval_sec=refresh_sec, ocr_enabled=ocr_on
            )
        else:
            stream_status = client.stream_status()
            if stream_status.get("running"):
                stream_status = client.stop_stream()
    except Exception:
        stream_status = {}

    st.caption(
        f"backend loop: {'🟢 running' if stream_status.get('running') else '🔴 stopped'} · "
        f"OCR: {'on' if stream_status.get('ocr_enabled', True) else 'off'} · "
        f"inspections: {stream_status.get('inspection_count', 0)} · "
        f"{stream_status.get('last_error') or 'ok'}"
    )

    if not continuous and st.button("▶️ Run one inspection", type="primary"):
        with st.spinner("Running inspection…"):
            try:
                client.inspect_live(ocr_enabled=ocr_on)
            except Exception as exc:
                st.error(f"inspection failed: {exc}")

    # Fetch ONLY the two processed frames from the backend.
    try:
        proc = client.processed_stream()
    except Exception as exc:
        proc = None
        st.error(f"could not fetch processed stream: {exc}")

    if proc is None:
        st.info(
            "Waiting for both phones to stream… point both phone apps at "
            f"`{backend_url}/upload`."
        )
    else:
        passed = proc.get("overall_pass")
        badge = "✅ PASS" if passed else ("❌ FAIL" if passed is not None else "…")
        st.subheader(f"Processed streams — {badge}")

        col_v, col_h = st.columns(2)
        if proc.get("annotated_vertical_b64"):
            show_image_bytes(
                col_v,
                base64.b64decode(proc["annotated_vertical_b64"]),
                f"Vertical camera ({proc.get('vertical_device_id') or '-'}) · "
                f"{proc.get('vertical_fps', 0)} FPS",
            )
        if proc.get("annotated_horizontal_b64"):
            show_image_bytes(
                col_h,
                base64.b64decode(proc["annotated_horizontal_b64"]),
                f"Horizontal camera ({proc.get('horizontal_device_id') or '-'}) · "
                f"{proc.get('horizontal_fps', 0)} FPS",
            )

        if not passed and proc.get("failure_reasons"):
            for reason in proc["failure_reasons"]:
                st.markdown(f"- 🔴 {reason}")
        st.caption(f"processing: {proc.get('total_time_ms', 0):.0f} ms")

    if continuous:
        time.sleep(refresh_sec)
        st.rerun()

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
                    vertical_file.getvalue(),
                    horizontal_file.getvalue(),
                    ocr_enabled=ocr_on,
                )
            except Exception as exc:
                st.error(f"inspection failed: {exc}")

    result = st.session_state.get("result")
    if result:
        st.divider()
        st.caption(f"Inspection `{result['inspection_id']}` · {result['created_at']}")
        render_full_result(result)
    else:
        st.info("Upload both images and run an inspection.")
