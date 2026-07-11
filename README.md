# Computer-Vision Quality Analysis System

End-to-end product quality inspection using **two mobile cameras as a stereo
rig**. Combines OCR, classical computer vision, stereo 3D measurement and
YOLO segmentation into one explainable **PASS / FAIL** decision shown on a
Streamlit dashboard.

```
Two Android phones (QC_Hackathon.apk)
        │  POST JPEG frames
        ▼
┌───────────────────┐   pair V/H     ┌────────────────────────────┐
│  backend :5000    │ ────────────► │  datascience :8100         │
│  (FastAPI)        │  POST /process│  (FastAPI)                 │
│  upload / pairing │ ◄──────────── │  OCR · 2D dims · stereo 3D │
│  orchestration    │  result JSON  │  defects · pothole · rules │
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
    processing_config.yaml  ROI, scale, OCR, stereo, defect, pothole settings
    product_specs.yaml      dimensions/tolerances + quality rules
    calibration/            saved calibration .npz files
  preprocessing/            resize, denoise, ROI, undistort, rectify
  calibration/              chessboard + stereo calibration scripts
  ocr/                      Sarvam AI OCR, field parsing, QR, validation
  dimensions_2d/            boundary extraction, geometric fitting, measures
  dimensions_3d/            disparity, depth reconstruction, 3D measures
  surface_defects/          YOLO detection: crack, dent, missing-head, paint-off, scratch
  pothole/                  YOLO segmentation + metrics (potholes ONLY)
  quality/                  rule engine + explainable decision
  overlays/                 dashboard overlay rendering
  models/                   yolo_model_final.pt (defects), pothole_yolov8_seg.pt (potholes)

start_all_services.py     one-command coordinator (starts + monitors all 3)
```

## What each inspection does

1. **Preprocessing** — resize, denoise, illumination normalization,
   undistortion + stereo rectification (when calibrated), configured ROI crop.
2. **OCR** (Sarvam AI Document Intelligence) — expiry date, manufacturing
   date, serial number, batch number, product ID + local QR decoding.
   Runs in parallel with the CV stages.
3. **2D dimensions** (classical CV only — no ML): threshold + Canny/Sobel +
   morphology + contours → length, width, diameter, radius, hole diameters,
   hole spacing, angles, area, perimeter, roundness.
4. **Stereo 3D** — SGBM disparity + WLS filter → depth map → height, depth,
   surface deformation, volume estimate.
5. **Surface defects** (YOLO object detection, `yolo_model_final.pt`) — one
   model localizes all five defect classes as bounding boxes: **Crack, Dent,
   Missing-head, Paint-off, Scratch**. Each detection reports class,
   confidence, bbox, and bbox-derived width/height/area (no masks — this
   model is detection-only).
6. **Pothole** — YOLO segmentation (only used for potholes): mask, bbox,
   confidence, area, perimeter, max width, severity.
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
| datascience | `datascience/.env.example` | `SARVAM_API_KEY` (required for OCR), `DATASCIENCE_HOST/PORT`, `POTHOLE_WEIGHTS_PATH`, `MM_PER_PX` |
| frontend | `frontend/.env.example` | `BACKEND_URL` |

## Configuration

| File | What you edit there |
|---|---|
| `backend/config.yaml` | ports, phone device-id → vertical/horizontal mapping, pair tolerance, continuous mode |
| `datascience/config/processing_config.yaml` | ROI box, mm/px scale, OCR key/language, stereo params, defect thresholds, YOLO weights path |
| `datascience/config/product_specs.yaml` | expected dimensions + tolerances, required OCR fields, per-class defect pass/fail rules (allowed / max count) — the whole quality rulebook |

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
measurements (height, depth, deformation, volume). For 2D-only real units
without full calibration, measure a reference object once and set
`scale.mm_per_px` in `processing_config.yaml` — this also converts surface
defect bounding boxes from pixels to millimeters.

## Surface defect detection (YOLO)

`datascience/models/yolo_model_final.pt` is a YOLO object-detection model
(bounding boxes, not masks) trained on five classes: `Crack`, `Dent`,
`Missing-head`, `Paint-off`, `Scratch`. It replaces the earlier classical-CV
crack/scratch/dent detectors as the sole surface-defect stage.

- Model path + confidence threshold: `defect_detection` in
  `datascience/config/processing_config.yaml`.
- Pass/fail per class (allow/deny, max count, confidence floor):
  `defect_rules` in `datascience/config/product_specs.yaml`.
- If the weights file is missing or `ultralytics` isn't installed, the
  surface-defects section reports `NOT_AVAILABLE` without affecting the rest
  of the inspection.

## Enabling pothole detection

Place YOLOv8 segmentation weights trained for potholes at
`datascience/models/pothole_yolov8_seg.pt` (or change
`pothole.weights_path`). Until then the pothole section reports
`NOT_AVAILABLE` without affecting the rest of the system.

## OCR (Sarvam AI)

OCR uses the Sarvam Document Intelligence job API. Put the key in
`datascience/.env` as `SARVAM_API_KEY` (see `datascience/.env.example`).
If it is missing, OCR reports `NOT_AVAILABLE` and the rest of the
inspection still runs.

## Troubleshooting

- **"need two streaming devices"** — both phones must have posted at least
  one frame to `/upload`; check they target the PC's LAN IP, port 5000.
- **Slow first pothole call** — YOLO loads lazily on the first inspection.
- **`ximgproc` missing** — install `opencv-contrib-python`
  (not plain `opencv-python`); it provides skeleton thinning + WLS filtering.
- **Windows firewall** — allow inbound port 5000 so the phones can reach
  the backend.
