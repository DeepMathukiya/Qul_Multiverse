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
    qr_present: bool = False
    qr_data: Optional[str] = None
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
# Surface defects
# --------------------------------------------------

class CrackMetrics(BaseModel):
    length: float
    max_width: float
    avg_width: float
    area: float
    orientation_deg: float
    branch_count: int
    unit: str = "px"


class ScratchMetrics(BaseModel):
    length: float
    width: float
    area: float
    orientation_deg: float
    unit: str = "px"


class DentMetrics(BaseModel):
    area: float
    diameter: float
    max_depth: float
    deformation: float
    unit: str = "mm"


class SurfaceDefectResult(BaseModel):
    status: CheckStatus = CheckStatus.NOT_AVAILABLE
    cracks: list[CrackMetrics] = Field(default_factory=list)
    scratches: list[ScratchMetrics] = Field(default_factory=list)
    dents: list[DentMetrics] = Field(default_factory=list)
    dent_note: Optional[str] = None
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
    overall_pass: bool
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
    # vertical, horizontal, roi, boundary, dimensions, depth_map, crack, scratch,
    # dent, pothole
    images: dict[str, str] = Field(default_factory=dict)

    timings_ms: dict[str, float] = Field(default_factory=dict)
    total_time_ms: float = 0.0


# --------------------------------------------------
# Service request/response payloads
# --------------------------------------------------

class ProcessRequest(BaseModel):
    """Backend -> datascience /process payload."""
    vertical_image_b64: str
    horizontal_image_b64: str
    vertical_device_id: Optional[str] = None
    horizontal_device_id: Optional[str] = None
