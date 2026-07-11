"""Backend FastAPI service — acquisition + orchestration layer.

Run:
    uvicorn backend.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI

# Load backend/.env before anything reads the environment
# (BACKEND_HOST/PORT, DATASCIENCE_URL, camera role ids).
load_dotenv(Path(__file__).resolve().parent / ".env")

from backend.routes.inspection_routes import router as inspection_router  # noqa: E402
from backend.routes.stream_routes import router as stream_router
from backend.routes.upload_routes import router as upload_router

app = FastAPI(
    title="QA Backend",
    description=(
        "Receives frames from the two mobile cameras, pairs them into stereo "
        "inspection events and delegates processing to the datascience service."
    ),
)

app.include_router(upload_router, tags=["acquisition"])
app.include_router(stream_router, tags=["stream"])
app.include_router(inspection_router, tags=["inspection"])


@app.get("/health")
def health():
    return {"status": "ok", "service": "backend"}


@app.on_event("startup")
def _start_continuous_mode():
    from backend.services.continuous_inspection import autostart_if_configured

    autostart_if_configured()


if __name__ == "__main__":
    import uvicorn

    from backend.config_loader import load_backend_config

    server = load_backend_config()["server"]
    uvicorn.run(
        "backend.main:app",
        host=server.get("host", "0.0.0.0"),
        port=int(server.get("port", 5000)),
    )
