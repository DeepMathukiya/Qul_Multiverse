"""Datascience FastAPI service — all main processing behind POST /process.

Run:
    uvicorn datascience.main:app --host 127.0.0.1 --port 8100
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI

# Load datascience/.env before anything reads the environment
# (SARVAM_API_KEY, DATASCIENCE_HOST/PORT, optional overrides).
load_dotenv(Path(__file__).resolve().parent / ".env")

from datascience.routes.process_routes import router as process_router  # noqa: E402

app = FastAPI(
    title="QA Datascience Service",
    description=(
        "Computer-vision quality analysis: preprocessing, OCR (Sarvam), 2D "
        "dimensions, stereo 3D, surface defects, YOLO pothole segmentation "
        "and quality-rule evaluation."
    ),
)

app.include_router(process_router, tags=["processing"])


@app.get("/health")
def health():
    return {"status": "ok", "service": "datascience"}


if __name__ == "__main__":
    import os

    import uvicorn

    uvicorn.run(
        "datascience.main:app",
        host=os.environ.get("DATASCIENCE_HOST", "127.0.0.1"),
        port=int(os.environ.get("DATASCIENCE_PORT", 8100)),
    )
