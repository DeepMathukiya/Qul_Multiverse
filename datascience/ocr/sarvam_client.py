"""Sarvam AI Document Intelligence client used for OCR.

Flow: BGR image -> temp PDF -> create job -> upload -> start ->
wait_until_complete -> download output ZIP -> extract text from the HTML.

Returns plain text; any failure (missing key, network, SDK) is reported as
an error string so the pipeline can degrade gracefully.
"""

from __future__ import annotations

import os
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from datascience.config_loader import load_system_config


def _image_to_pdf(image: np.ndarray, pdf_path: str) -> None:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(pdf_path, "PDF", resolution=150.0)


def _text_from_output_zip(zip_path: str) -> str:
    """Extract readable text from every HTML/TXT file in the job output ZIP."""
    from bs4 import BeautifulSoup

    texts: list[str] = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            lower = name.lower()
            if lower.endswith((".html", ".htm")):
                soup = BeautifulSoup(zf.read(name).decode("utf-8", "ignore"), "html.parser")
                texts.append(soup.get_text(separator="\n"))
            elif lower.endswith((".txt", ".md")):
                texts.append(zf.read(name).decode("utf-8", "ignore"))
    return "\n".join(t.strip() for t in texts if t.strip())


def extract_text(image: np.ndarray) -> tuple[str, str | None]:
    """Run Sarvam OCR on one image. Returns (text, error)."""
    ocr_cfg = load_system_config().get("ocr", {})
    api_key = (
        os.environ.get(ocr_cfg.get("api_key_env", "SARVAM_API_KEY"), "")
        or ocr_cfg.get("api_key")
        or ""
    )

    if not api_key:
        return "", (
            f"Sarvam API key not set (env {ocr_cfg.get('api_key_env', 'SARVAM_API_KEY')} "
            "or 'api_key' in processing_config.yaml)"
        )

    try:
        from sarvamai import SarvamAI
    except ImportError:
        return "", "sarvamai package not installed (pip install sarvamai)"

    tmpdir = tempfile.mkdtemp(prefix="sarvam_ocr_")
    pdf_path = str(Path(tmpdir) / "inspection.pdf")
    zip_path = str(Path(tmpdir) / "output.zip")

    try:
        _image_to_pdf(image, pdf_path)

        client = SarvamAI(api_subscription_key=api_key)

        job = client.document_intelligence.create_job(
            language=ocr_cfg.get("language", "en-IN"),
            output_format="html",
        )
        job.upload_file(pdf_path)
        job.start()

        status = job.wait_until_complete()
        state = getattr(status, "job_state", "unknown")
        if str(state).lower() not in ("completed", "success", "succeeded", "done"):
            return "", f"Sarvam job ended in state '{state}'"

        job.download_output(zip_path)
        text = _text_from_output_zip(zip_path)
        if not text:
            return "", "Sarvam job produced no text output"
        return text, None

    except Exception as exc:  # network/SDK errors must not kill the pipeline
        return "", f"Sarvam OCR failed: {exc}"
