"""Streamlit rendering of one InspectionResult (dict form from the API)."""

from __future__ import annotations

import base64
import inspect

import streamlit as st

# st.image's full-width kwarg differs across Streamlit versions:
# <1.31 / 1.33: use_column_width, newer: use_container_width.
_IMAGE_WIDTH_KWARGS = (
    {"use_container_width": True}
    if "use_container_width" in inspect.signature(st.image).parameters
    else {"use_column_width": True}
)

_STATUS_ICONS = {
    "PASS": "✅",
    "FAIL": "❌",
    "NOT_AVAILABLE": "⚪",
    "SKIPPED": "➖",
}


def _b64_to_bytes(b64: str) -> bytes:
    return base64.b64decode(b64)


def _show_image(images: dict, key: str, caption: str, container=st) -> bool:
    if key in images:
        container.image(
            _b64_to_bytes(images[key]), caption=caption, **_IMAGE_WIDTH_KWARGS
        )
        return True
    return False


def show_image_bytes(container, data: bytes, caption: str) -> None:
    """Version-safe full-width image rendering (also used by the dashboard)."""
    container.image(data, caption=caption, **_IMAGE_WIDTH_KWARGS)


def render_decision_banner(result: dict) -> None:
    quality = result.get("quality") or {}
    if quality.get("overall_pass"):
        st.success("## ✅ PASS — product accepted")
    else:
        st.error("## ❌ FAIL — product rejected")
        for reason in quality.get("failure_reasons", []):
            st.markdown(f"- 🔴 {reason}")


def render_camera_images(result: dict) -> None:
    st.subheader("📷 Camera images")
    images = result.get("images", {})
    col_v, col_h = st.columns(2)
    _show_image(
        images, "vertical",
        f"Vertical camera ({result.get('vertical_device_id') or '-'})", col_v,
    )
    _show_image(
        images, "horizontal",
        f"Horizontal camera ({result.get('horizontal_device_id') or '-'})", col_h,
    )
    _show_image(images, "roi", "Product ROI (configured)")


def render_dimensions_2d(result: dict) -> None:
    dims = result.get("dims_2d", {})
    st.subheader("📏 2D dimensional analysis")
    st.caption(
        f"status: {_STATUS_ICONS.get(dims.get('status'), '')} {dims.get('status')} · "
        f"scale: {dims.get('scale_source')} "
        f"({dims.get('mm_per_px') or 'no mm/px — pixel units'})"
    )
    _show_image(result.get("images", {}), "boundary", "Detected boundary + dimensions")

    measurements = dims.get("measurements", [])
    if measurements:
        st.table(
            [{"measurement": m["name"], "value": m["value"], "unit": m["unit"]}
             for m in measurements]
        )
    elif dims.get("error"):
        st.warning(dims["error"])


def render_dimensions_3d(result: dict) -> None:
    dims = result.get("dims_3d", {})
    st.subheader("🧊 3D stereo analysis")
    st.caption(
        f"status: {_STATUS_ICONS.get(dims.get('status'), '')} {dims.get('status')} · "
        f"calibrated: {dims.get('calibrated')}"
    )
    _show_image(result.get("images", {}), "depth_map", "Stereo depth map")

    measurements = dims.get("measurements", [])
    if measurements:
        st.table(
            [{"measurement": m["name"], "value": m["value"], "unit": m["unit"]}
             for m in measurements]
        )
    if dims.get("note"):
        st.info(dims["note"])
    if dims.get("error"):
        st.warning(dims["error"])


def render_ocr(result: dict) -> None:
    ocr = result.get("ocr", {})
    st.subheader("🔤 OCR & product information")
    st.caption(f"status: {_STATUS_ICONS.get(ocr.get('status'), '')} {ocr.get('status')} "
               f"· provider: {ocr.get('provider')}")

    fields = ocr.get("fields", {})
    st.table(
        [
            {"field": "Expiry date", "value": fields.get("expiry_date") or "—",
             "note": "valid" if ocr.get("expiry_valid") else
             ("EXPIRED" if ocr.get("expiry_valid") is False else "unverified")},
            {"field": "Manufacturing date", "value": fields.get("manufacturing_date") or "—", "note": ""},
            {"field": "Serial number", "value": fields.get("serial_number") or "—", "note": ""},
            {"field": "Batch number", "value": fields.get("batch_number") or "—", "note": ""},
            {"field": "Product ID", "value": fields.get("product_id") or "—", "note": ""},
            {"field": "QR code", "value": ocr.get("qr_data") or "not detected",
             "note": "present" if ocr.get("qr_present") else "missing"},
        ]
    )
    if ocr.get("missing_required_fields"):
        st.warning("Missing required fields: " + ", ".join(ocr["missing_required_fields"]))
    if ocr.get("error"):
        st.warning(ocr["error"])
    if ocr.get("raw_text"):
        with st.expander("Raw OCR text"):
            st.text(ocr["raw_text"])


def render_surface_defects(result: dict) -> None:
    defects = result.get("surface_defects", {})
    images = result.get("images", {})
    st.subheader("🛠️ Metal surface defects")
    st.caption(f"status: {_STATUS_ICONS.get(defects.get('status'), '')} {defects.get('status')}")

    col_c, col_s, col_d = st.columns(3)
    col_c.metric("Cracks", len(defects.get("cracks", [])))
    col_s.metric("Scratches", len(defects.get("scratches", [])))
    col_d.metric("Dents", len(defects.get("dents", [])))

    if defects.get("cracks"):
        _show_image(images, "crack", "Crack detections")
        st.table([
            {"crack": i + 1, f"length ({c['unit']})": c["length"],
             f"max width ({c['unit']})": c["max_width"],
             f"avg width ({c['unit']})": c["avg_width"],
             f"area ({c['unit']}²)": c["area"],
             "orientation °": c["orientation_deg"], "branches": c["branch_count"]}
            for i, c in enumerate(defects["cracks"])
        ])

    if defects.get("scratches"):
        _show_image(images, "scratch", "Scratch detections")
        st.table([
            {"scratch": i + 1, f"length ({s['unit']})": s["length"],
             f"width ({s['unit']})": s["width"], f"area ({s['unit']}²)": s["area"],
             "orientation °": s["orientation_deg"]}
            for i, s in enumerate(defects["scratches"])
        ])

    if defects.get("dents"):
        _show_image(images, "dent", "Dent regions (from stereo depth)")
        st.table([
            {"dent": i + 1, "area (mm²)": d["area"], "diameter (mm)": d["diameter"],
             "max depth (mm)": d["max_depth"], "deformation RMS (mm)": d["deformation"]}
            for i, d in enumerate(defects["dents"])
        ])
    elif defects.get("dent_note"):
        st.info(defects["dent_note"])

    if defects.get("error"):
        st.warning(defects["error"])


def render_pothole(result: dict) -> None:
    pothole = result.get("pothole", {})
    if not pothole.get("enabled"):
        return
    st.subheader("🕳️ Pothole analysis (YOLO segmentation)")
    st.caption(f"status: {_STATUS_ICONS.get(pothole.get('status'), '')} {pothole.get('status')}")

    _show_image(result.get("images", {}), "pothole", "Pothole segmentation")

    if pothole.get("detections"):
        st.table([
            {"pothole": i + 1, "confidence": d["confidence"], "severity": d["severity"],
             f"area ({d['unit']}²)": d["area"], f"perimeter ({d['unit']})": d["perimeter"],
             f"max width ({d['unit']})": d["max_width"]}
            for i, d in enumerate(pothole["detections"])
        ])
    elif pothole.get("status") == "PASS":
        st.success("No potholes detected")

    if pothole.get("note"):
        st.caption(pothole["note"])
    if pothole.get("error"):
        st.warning(pothole["error"])


def render_quality_checks(result: dict) -> None:
    quality = result.get("quality") or {}
    st.subheader("📋 Quality checks")
    checks = quality.get("checks", [])
    if not checks:
        st.info("no checks evaluated")
        return
    st.table([
        {
            "": _STATUS_ICONS.get(c["status"], ""),
            "check": c["name"],
            "measured": c.get("measured") or "—",
            "expected": c.get("expected") or "—",
            "status": c["status"],
            "reason": c.get("reason") or "",
        }
        for c in checks
    ])


def render_timings(result: dict) -> None:
    st.subheader("⏱️ Processing time")
    timings = result.get("timings_ms", {})
    if timings:
        st.table([{"stage": k, "ms": v} for k, v in timings.items()])
    st.caption(f"total: {result.get('total_time_ms', 0):.0f} ms")


def render_full_result(result: dict) -> None:
    """Main view: everything is drawn ON the stream frame itself.

    The classic section-by-section report stays available in an expander.
    """
    render_decision_banner(result)

    images = result.get("images", {})
    if "annotated" in images:
        _show_image(images, "annotated", "Inspection stream (all results overlaid)")
    else:
        # Older results without the composite frame fall back to raw cameras.
        render_camera_images(result)

    quality = result.get("quality") or {}
    total = len(quality.get("checks", []))
    failed = sum(1 for c in quality.get("checks", []) if c["status"] == "FAIL")
    st.caption(
        f"checks: {total - failed}/{total} passed · "
        f"processing: {result.get('total_time_ms', 0):.0f} ms"
    )

    with st.expander("📋 Detailed report"):
        render_camera_images(result)
        st.divider()
        render_ocr(result)
        st.divider()
        render_dimensions_2d(result)
        st.divider()
        render_dimensions_3d(result)
        st.divider()
        render_surface_defects(result)
        st.divider()
        render_pothole(result)
        st.divider()
        render_quality_checks(result)
        render_timings(result)
