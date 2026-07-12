"""Pydantic models owned by the datascience layer (its API contract).

Every physical measurement carries its unit explicitly. Values in mm exist
only when a valid calibration / reference scale was available — otherwise
the stage reports status NOT_AVAILABLE instead of faking numbers.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class CheckStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_AVAILABLE = "NOT_AVAILABLE"
    SKIPPED = "SKIPPED"


# --------------------------------------------------
# OCR
# --------------------------------------------------

class OcrFields(BaseModel):
    expiry_date: Optional[str] = None
    manufacturing_date: Optional[str] = None
    serial_number: Optional[str] = None
    batch_number: Optional[str] = None
    product_id: Optional[str] = None


class OcrResult(BaseModel):
    status: CheckStatus = CheckStatus.NOT_AVAILABLE
    provider: str = "sarvam"
    raw_text: str = ""
    fields: OcrFields = Field(default_factory=OcrFields)
    expiry_valid: Optional[bool] = None
    missing_required_fields: list[str] = Field(default_factory=list)
    error: Optional[str] = None


# --------------------------------------------------
# 2D dimensions
# --------------------------------------------------

class Measurement(BaseModel):
    name: str
    value: float
    unit: str  # "mm", "deg", "px", "mm2", "ratio", ...


class Dim2DResult(BaseModel):
    status: CheckStatus = CheckStatus.NOT_AVAILABLE
    scale_source: str = "none"  # "calibration" | "reference_object" | "config" | "none"
    mm_per_px: Optional[float] = None
    measurements: list[Measurement] = Field(default_factory=list)
    boundary_found: bool = False
    error: Optional[str] = None


# --------------------------------------------------
# 3D dimensions
# --------------------------------------------------

class Dim3DResult(BaseModel):
    status: CheckStatus = CheckStatus.NOT_AVAILABLE
    calibrated: bool = False
    measurements: list[Measurement] = Field(default_factory=list)
    note: Optional[str] = None
    error: Optional[str] = None


# --------------------------------------------------
# Surface defects (YOLO object detection — bounding boxes only)
# --------------------------------------------------

class DefectDetection(BaseModel):
    """One YOLO-detected surface defect. Geometry is bbox-derived only
    (this model outputs boxes, not masks). Real-world (mm) size is filled in
    only when a valid scale exists — otherwise values stay in pixels."""
    label: str                        # Crack | Dent | Missing-head | Paint-off | Scratch
    confidence: float
    bbox_xyxy: list[float]            # [x1, y1, x2, y2] in pixels
    width: float                      # bbox width  (unit below)
    height: float                     # bbox height (unit below)
    area: float                       # bbox area   (unit below, squared)
    unit: str = "px"                 # "px" | "mm"


class SurfaceDefectResult(BaseModel):
    status: CheckStatus = CheckStatus.NOT_AVAILABLE
    enabled: bool = False
    present: bool = False
    detections: list[DefectDetection] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)  # per-class detection counts
    note: Optional[str] = None
    error: Optional[str] = None


# --------------------------------------------------
# Pothole (YOLO segmentation only)
# --------------------------------------------------

class PotholeDetection(BaseModel):
    confidence: float
    bbox_xyxy: list[float]
    area: float
    perimeter: float
    max_width: float
    unit: str = "px"
    severity: str = "low"  # low | medium | high


class PotholeResult(BaseModel):
    status: CheckStatus = CheckStatus.NOT_AVAILABLE
    enabled: bool = False
    present: bool = False
    detections: list[PotholeDetection] = Field(default_factory=list)
    note: Optional[str] = None
    error: Optional[str] = None


# --------------------------------------------------
# Quality decision
# --------------------------------------------------

class QualityCheck(BaseModel):
    name: str
    status: CheckStatus
    measured: Optional[str] = None
    expected: Optional[str] = None
    reason: Optional[str] = None


class QualityDecision(BaseModel):
    # None = no judgement could be made (nothing in the frame(s) was
    # verifiable — every check came back NOT_AVAILABLE/SKIPPED).
    overall_pass: Optional[bool] = None
    checks: list[QualityCheck] = Field(default_factory=list)
    failure_reasons: list[str] = Field(default_factory=list)


# --------------------------------------------------
# Full inspection result
# --------------------------------------------------

class InspectionResult(BaseModel):
    inspection_id: str
    created_at: str
    vertical_device_id: Optional[str] = None
    horizontal_device_id: Optional[str] = None

    ocr: OcrResult = Field(default_factory=OcrResult)
    dims_2d: Dim2DResult = Field(default_factory=Dim2DResult)
    dims_3d: Dim3DResult = Field(default_factory=Dim3DResult)
    surface_defects: SurfaceDefectResult = Field(default_factory=SurfaceDefectResult)
    pothole: PotholeResult = Field(default_factory=PotholeResult)
    quality: Optional[QualityDecision] = None

    # base64-encoded JPEG/PNG images keyed by name:
    # vertical, horizontal, roi, boundary, dimensions, depth_map, defects, pothole,
    # annotated_vertical, annotated_horizontal
    images: dict[str, str] = Field(default_factory=dict)

    timings_ms: dict[str, float] = Field(default_factory=dict)
    total_time_ms: float = 0.0


# --------------------------------------------------
# Service request/response payloads
# --------------------------------------------------

class ProcessRequest(BaseModel):
    """Backend -> datascience /process payload.

    Both images present = full stereo inspection (dimensions + volume +
    defects). Only one present = single-camera fallback (no stereo pair to
    measure area/volume from) — the product is judged on surface-defect
    (crack/dent) detection alone.
    """
    vertical_image_b64: Optional[str] = None
    horizontal_image_b64: Optional[str] = None
    vertical_device_id: Optional[str] = None
    horizontal_device_id: Optional[str] = None
    # None = use processing_config.yaml; True/False = per-request override.
    ocr_enabled: Optional[bool] = None
    # True = this pair came from Upload mode — its measured area becomes the
    # new reference baseline for area_reference_rules (see reference_store.py).
    is_upload: bool = False
    # None = use product_specs.yaml's area_reference_rules.tolerance_ratio;
    # otherwise a per-request override (e.g. the dashboard's tolerance slider).
    area_tolerance_ratio: Optional[float] = None
