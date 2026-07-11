# Computer-Vision Quality Analysis System

End-to-end product quality inspection using **two mobile cameras as a stereo
rig**. Combines classical computer vision (2D dimensions, stereo 3D
measurement, crack/scratch/dent detection) with **Qwen3-VL-4B-Instruct**
(served locally via Qualcomm GenieX, on-NPU) for OCR field extraction and
pothole detection, into one explainable **PASS / FAIL** decision shown on a
Streamlit dashboard.

```
Two Android phones (QC_Hackathon.apk)
        │  POST JPEG frames
        ▼
┌───────────────────┐   pair V/H     ┌────────────────────────────┐
│  backend :5000    │ ────────────► │  datascience :8100         │
│  (FastAPI)        │  POST /process│  (FastAPI)                 │
│  upload / pairing │ ◄──────────── │  Qwen3-VL OCR+pothole ·    │
│  orchestration    │  result JSON  │  2D/3D dims · defects · rules │
└───────────────────┘               └────────────────────────────┘
        ▲  REST
        │
┌───────────────────┐
│  frontend :8501   │
│  (Streamlit)      │
└───────────────────┘
```

The three layers are **fully independent** — each has its own code, config
and helpers. Nothing is shared between them; they talk only over HTTP.

## Quick start

```bash
pip install -r requirements.txt

# start everything (datascience -> backend -> frontend, health-checked):
python start_all_services.py
```

Then:

- **Dashboard**: open http://127.0.0.1:8501
- **Phones**: point the QC_Hackathon app of *both* phones at
  `http://<this-pc-ip>:5000/upload` (same address the original
  `example_receiver.py` used)
- **API docs**: http://127.0.0.1:5000/docs (backend) ·
  http://127.0.0.1:8100/docs (datascience)

You can also start each layer manually:

```bash
python -m uvicorn datascience.main:app --port 8100
python -m uvicorn backend.main:app --host 0.0.0.0 --port 5000
python -m streamlit run frontend/streamlit_dashboard.py
```

No phones handy? Use the dashboard's **Upload images** mode with any two
photos, or run one inspection from the command line:

```bash
python -m datascience.inspection_pipeline --vertical v.jpg --horizontal h.jpg
```

## Project layout

```
frontend/                 Streamlit UI (display only, REST client)
  streamlit_dashboard.py    entry point
  api_client.py             backend HTTP client
  result_views.py           result rendering sections

backend/                  FastAPI acquisition + orchestration (port 5000)
  main.py                   entry point
  config.yaml               ports, camera roles, datascience URL
  routes/                   /upload, /latest_pair, /inspect, /inspections
  services/                 frame store, V/H pairing, datascience client

datascience/              FastAPI processing service (port 8100)
  main.py                   entry point
  inspection_pipeline.py    orchestrates all stages (also a CLI)
  config/
    processing_config.yaml  ROI, scale, GenieX/VLM endpoint, stereo, defect settings
    product_specs.yaml      dimensions/tolerances + quality rules
    calibration/            saved calibration .npz files
  preprocessing/            resize, denoise, ROI, undistort, rectify
  calibration/              chessboard + stereo calibration scripts
  vlm/                      shared GenieX client, prompts, JSON parsing
  ocr/                      Qwen3-VL field extraction, QR decode, expiry validation
  dimensions_2d/            boundary extraction, geometric fitting, measures
  dimensions_3d/            disparity, depth reconstruction, 3D measures
  surface_defects/          crack / scratch (classical CV), dent (stereo)
  pothole/                  Qwen3-VL qualitative detection (potholes ONLY)
  quality/                  rule engine + explainable decision
  overlays/                 dashboard overlay rendering

start_all_services.py     one-command coordinator (starts + monitors all 3)
```

## What each inspection does

1. **Preprocessing** — resize, denoise, illumination normalization,
   undistortion + stereo rectification (when calibrated), configured ROI crop.
2. **OCR** (Qwen3-VL-4B-Instruct via Qualcomm GenieX, local NPU inference) —
   expiry date, manufacturing date, serial number, batch number, product ID
   + local QR decoding (deterministic, always runs). Runs in a background
   thread alongside the CV stages.
3. **2D dimensions** (classical CV only — no ML): threshold + Canny/Sobel +
   morphology + contours → length, width, diameter, radius, hole diameters,
   hole spacing, angles, area, perimeter, roundness.
4. **Stereo 3D** — SGBM disparity + WLS filter → depth map → height, depth,
   surface deformation, volume estimate.
5. **Surface defects** (classical CV): cracks (length, max/avg width, area,
   orientation, branches), scratches (length, width, area, orientation),
   dents from stereo depth (area, diameter, max depth, deformation).
6. **Pothole** — Qwen3-VL-4B-Instruct (same GenieX call path as OCR, runs
   sequentially after it in the same background thread): qualitative
   presence, severity (low/medium/high) and a short description per
   detection. No pixel mask or geometry (bbox/area/perimeter) — a VLM can't
   ground precise coordinates the way the previous YOLO segmentation did.
7. **Quality rules** — every configured rule becomes a check with measured
   value, expected range and reason → explainable PASS/FAIL.

### Honest units (important)

Millimeter values are reported **only** when a valid calibration or measured
reference scale exists. Without it, measurements stay in pixels and the
corresponding tolerance checks report `NOT_AVAILABLE` instead of comparing
pixels against millimeter specs.

## Continuous streaming mode

The backend runs a background loop that keeps inspecting the newest
vertical/horizontal pair automatically (`continuous.enabled: true` in
`backend/config.yaml`, on by default). The dashboard's **Live** mode has a
🔁 toggle that starts/stops this loop and auto-refreshes the page, so
results flow in continuously while the phones stream. Runtime control:

```
POST /stream/start?interval_sec=5   POST /stream/stop   GET /stream/status
```

## Environment variables (.env)

Each layer loads its **own** `.env` file via `python-dotenv` — copy the
`.env.example` next to it and fill in values (env vars override the YAML):

| Layer | File | Key variables |
|---|---|---|
| backend | `backend/.env.example` | `BACKEND_HOST/PORT`, `DATASCIENCE_URL`, `VERTICAL/HORIZONTAL_DEVICE_ID`, `CONTINUOUS_ENABLED`, `CONTINUOUS_INTERVAL_SEC` |
| datascience | `datascience/.env.example` | `GENIEX_BASE_URL`/`GENIEX_MODEL`/`GENIEX_API_KEY` (required for OCR + pothole), `DATASCIENCE_HOST/PORT`, `MM_PER_PX` |
| frontend | `frontend/.env.example` | `BACKEND_URL` |

## Configuration

| File | What you edit there |
|---|---|
| `backend/config.yaml` | ports, phone device-id → vertical/horizontal mapping, pair tolerance, continuous mode |
| `datascience/config/processing_config.yaml` | ROI box, mm/px scale, GenieX base URL/model/timeout for OCR+pothole, stereo params, defect thresholds |
| `datascience/config/product_specs.yaml` | expected dimensions + tolerances, required OCR fields, crack/scratch/dent limits — the whole quality rulebook |

Quality rules are pure data — tune tolerances without touching code.

## Calibration (unlocks 3D + real-world units)

```bash
# 1. intrinsics per phone (15-25 chessboard photos each)
python -m datascience.calibration.camera_calibration --images calib/vertical  --camera vertical
python -m datascience.calibration.camera_calibration --images calib/horizontal --camera horizontal

# 2. stereo calibration (synchronized chessboard pairs, same filenames)
python -m datascience.calibration.stereo_calibration --vertical calib/stereo/vertical --horizontal calib/stereo/horizontal
```

After this, the 3D section switches from `NOT_AVAILABLE` to metric
measurements and dent-depth checks activate. For 2D-only real units without
full calibration, measure a reference object once and set `scale.mm_per_px`
in `processing_config.yaml`.

## OCR & pothole detection (Qwen3-VL via GenieX)

Both stages call the same locally-hosted **Qwen3-VL-4B-Instruct** model
through Qualcomm **GenieX**'s OpenAI-compatible server, running on-NPU
(tested target: Snapdragon X Elite). Setup (commands sourced from Qualcomm's
GenieX documentation — verify against your installed GenieX version):

```bash
# 1. install GenieX (Windows ARM64 installer), then open a new terminal
# 2. pull the pre-compiled NPU bundle
geniex pull ai-hub-models/Qwen3-VL-4B-Instruct
# 3. start the local OpenAI-compatible server (default http://127.0.0.1:18181/v1)
geniex serve
```

Point `datascience/.env` at it (see `datascience/.env.example`):
`GENIEX_BASE_URL`, `GENIEX_MODEL`, `GENIEX_API_KEY`. If GenieX isn't
reachable, OCR and pothole detection report `NOT_AVAILABLE` with an error
note — the rest of the inspection (2D dims, stereo 3D, surface defects)
still runs normally, since those stay classical CV and don't depend on it.

Pothole detection reports qualitative severity (low/medium/high) + a short
description per finding — no pixel mask or geometry, since a VLM can't
ground precise bounding boxes/contours the way the previous YOLO
segmentation did.

## Troubleshooting

- **"need two streaming devices"** — both phones must have posted at least
  one frame to `/upload`; check they target the PC's LAN IP, port 5000.
- **`ximgproc` missing** — install `opencv-contrib-python`
  (not plain `opencv-python`); it provides skeleton thinning + WLS filtering.
- **Windows firewall** — allow inbound port 5000 so the phones can reach
  the backend.
- **GenieX server not reachable** — verify `geniex serve` is running and
  `GENIEX_BASE_URL` in `datascience/.env` matches its printed address; OCR
  and pothole detection return `NOT_AVAILABLE` with an error note instead of
  crashing `/process`.
