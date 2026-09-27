"""ANPR service using the existing processor without fabricated plate text."""
from __future__ import annotations

from io import BytesIO
import importlib.util
import logging
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image, UnidentifiedImageError

from .model_runtime import (
    collect_released_memory,
    current_rss_mb,
    serialized_model_operation,
)

logger = logging.getLogger("uvicorn.error")
SERVICE_FILE = Path(__file__).resolve()
APP_ROOT = SERVICE_FILE.parents[1]
REPO_ROOT = SERVICE_FILE.parents[3]
EDGE_ANPR = APP_ROOT / "processors" / "anpr.py"

_spec = importlib.util.spec_from_file_location("urban_edge_anpr", EDGE_ANPR)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load ANPR processor from {EDGE_ANPR}")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
ANPRProcessor = _mod.ANPRProcessor


def _weight_candidates(filename: str) -> list[Path]:
    candidates: list[Path] = []
    configured = os.getenv("ANPR_MODEL_PATH", "").strip()
    if configured:
        candidates.append(Path(configured).expanduser())
    # Render builds the backend with backend/ as Docker build context, so this
    # backend-local copy must be first. The frontend path remains for local/full-repo runs.
    candidates.extend([
        APP_ROOT / "weights" / filename,
        REPO_ROOT / "frontend" / "ml" / "weights" / filename,
        Path.cwd() / "frontend" / "ml" / "weights" / filename,
    ])
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            unique.append(resolved)
    return unique


def _find_weight(filename: str) -> Path | None:
    return next((candidate for candidate in _weight_candidates(filename) if candidate.is_file()), None)


WEIGHTS_PT = _find_weight("anpr_plate.pt")
WEIGHTS_ONNX = _find_weight("anpr_plate.onnx")
WEIGHTS = WEIGHTS_PT or WEIGHTS_ONNX
_processor = None
_processor_error = None
_processor_load_ms = None


def _log_weight_diagnostics() -> None:
    logger.info("ANPR Ultralytics available: %s", importlib.util.find_spec("ultralytics") is not None)
    for candidate in _weight_candidates("anpr_plate.pt"):
        exists = candidate.is_file()
        size = candidate.stat().st_size if exists else None
        logger.info("ANPR model candidate: %s exists=%s size=%s", candidate, exists, size)


def get_processor():
    global _processor, _processor_error, _processor_load_ms
    if _processor is None:
        started = time.perf_counter()
        _log_weight_diagnostics()
        try:
            # Never retain the much larger road/traffic detector alongside
            # ANPR on a memory-constrained service instance.
            from .road_detector import release_road_models
            from .urban_vision import release_traffic_model

            release_road_models()
            release_traffic_model()
            _processor = ANPRProcessor(model_path=str(WEIGHTS) if WEIGHTS else None)
            if WEIGHTS and _processor.model is not None:
                _processor_error = None
                logger.info("ANPR trained plate detector loaded successfully: %s", WEIGHTS.name)
            elif WEIGHTS:
                detail = getattr(_processor, "model_load_error", None)
                _processor_error = detail or f"ANPR processor could not load {WEIGHTS.name}"
                logger.error("ANPR processor could not load model: path=%s reason=%s", WEIGHTS, _processor_error)
            else:
                _processor_error = "ANPR weight was not found in the deployed filesystem"
                logger.error("%s", _processor_error)
        except Exception as exc:
            _processor_error = f"ANPR model load failed: {type(exc).__name__}: {exc}"
            logger.exception("ANPR processor initialization failed")
            # Keep the API alive without pretending that a trained detector loaded.
            _processor = ANPRProcessor()
        _processor_load_ms = round((time.perf_counter() - started) * 1000, 2)
    return _processor


def warm_anpr_model():
    return get_processor()


def release_anpr_model() -> None:
    global _processor
    if _processor is None:
        return
    if hasattr(_processor, "_easyocr_reader"):
        _processor._easyocr_reader = None
    _processor.model = None
    _processor = None
    collect_released_memory()


def anpr_model_health():
    # Health checks must never allocate a YOLO/OCR model. The frontend calls
    # this before every upload, and loading here previously caused Render OOMs.
    ready = bool(WEIGHTS and _processor is not None and _processor.model is not None)
    return {
        "anpr_model_ready": ready,
        "anpr_weight_available": bool(WEIGHTS),
        "anpr_weight": WEIGHTS.name if WEIGHTS else None,
        "model_cached": _processor is not None,
        "anpr_error": _processor_error,
        "model_load_ms": _processor_load_ms,
        "ocr_cached": bool(_processor is not None and hasattr(_processor, "_tesseract_ready")),
        "process_rss_mb": current_rss_mb(),
    }


@serialized_model_operation
def recognize_plate(raw: bytes):
    request_started = time.perf_counter()
    decode_started = request_started
    try:
        with Image.open(BytesIO(raw)) as im:
            frame = np.array(im.convert("RGB"))[:, :, ::-1].copy()
    except (UnidentifiedImageError, OSError):
        raise ValueError("Invalid image")

    decode_ms = (time.perf_counter() - decode_started) * 1000
    processor = get_processor()
    detector_ready = bool(WEIGHTS and processor.model is not None)
    result = processor.process_vehicle_crop(0, frame, use_temporal_cache=False)

    if not detector_ready:
        status = "model_unavailable"
    elif result.localizer_status == "localized" and result.plate_number == "UNKNOWN":
        status = "plate_detected_ocr_failed"
    elif result.localizer_status == "localized":
        status = "plate_detected"
    else:
        status = "no_plate_detected"

    return {
        "plate_number": result.plate_number,
        "raw_ocr_text": result.raw_ocr_text,
        "plate_detection_confidence": result.plate_detection_confidence,
        "ocr_confidence": result.ocr_confidence,
        "overall_confidence": result.overall_confidence,
        "is_format_valid": result.is_format_valid,
        "requires_manual_verification": result.requires_manual_verification,
        "timestamp": result.timestamp,
        "plate_bbox": result.plate_bbox,
        "localizer_status": result.localizer_status,
        "anpr_status": status,
        "timing": {
            "image_decode_ms": round(decode_ms, 2),
            "model_load_ms": _processor_load_ms or 0.0,
            **result.timings,
            "total_ms": round((time.perf_counter() - request_started) * 1000, 2),
        },
        # Detector readiness is independent of OCR success. UNKNOWN must not
        # downgrade a successfully loaded trained localizer to prototype mode.
        "mode": "trained_plate_detector" if detector_ready else "ocr_prototype_no_plate_detector",
        "warning": None if detector_ready else (
            "Trained plate-localizer model is unavailable; OCR result must be manually verified."
        ),
    }
