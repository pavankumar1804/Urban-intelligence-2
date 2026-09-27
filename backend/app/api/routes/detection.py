from fastapi import APIRouter, File, HTTPException, UploadFile
from datetime import datetime, timezone
import time
from loguru import logger
from starlette.concurrency import run_in_threadpool

from app.services.road_detector import detect_road_defects, road_model_health
from app.services.anpr_service import anpr_model_health, recognize_plate
from app.services.urban_vision import (
    analyze_infrastructure,
    analyze_safety,
    detect_traffic as detect_traffic_frame,
    detection_health as urban_detection_health,
)


router = APIRouter(prefix="/api/detect", tags=["AI Detection"])

MAX_FILE_SIZE = 5 * 1024 * 1024


def _bbox_iou(a, b):
    """Return IoU for two API detection boxes."""
    ax1, ay1, ax2, ay2 = a["x1"], a["y1"], a["x2"], a["y2"]
    bx1, by1, bx2, by2 = b["x1"], b["y1"], b["x2"], b["y2"]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(1e-6, area_a + area_b - inter)


def _bbox_intersection_over_source(source, other):
    """Return the fraction of ``source`` covered by ``other``."""
    sx1, sy1, sx2, sy2 = source["x1"], source["y1"], source["x2"], source["y2"]
    ox1, oy1, ox2, oy2 = other["x1"], other["y1"], other["x2"], other["y2"]
    ix1, iy1 = max(sx1, ox1), max(sy1, oy1)
    ix2, iy2 = min(sx2, ox2), min(sy2, oy2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    source_area = max(0.0, sx2 - sx1) * max(0.0, sy2 - sy1)
    return intersection / max(1e-6, source_area)


def _prefer_model_over_water_heuristic(detections):
    """Keep YOLO evidence authoritative over prototype water heuristics.

    The OpenCV water detector is intentionally only fallback evidence.  It must
    not turn a region already identified as a pothole by the trained model into
    a competing waterlogging result.  Real model confidence values are left
    untouched.
    """
    model_detections = [
        d for d in detections
        if d.get("detection_method") != "COMPUTER-VISION PROTOTYPE"
    ]
    if not model_detections:
        return detections

    filtered = []
    for detection in detections:
        is_water_heuristic = (
            str(detection.get("class_name", "")).lower() == "waterlogging"
            and detection.get("detection_method") == "COMPUTER-VISION PROTOTYPE"
        )
        if is_water_heuristic:
            box = detection.get("bbox")
            conflicts_with_model = any(
                box and model_detection.get("bbox")
                and (
                    _bbox_iou(box, model_detection["bbox"]) >= 0.10
                    or _bbox_intersection_over_source(box, model_detection["bbox"]) >= 0.35
                )
                for model_detection in model_detections
            )
            if conflicts_with_model:
                continue
        filtered.append(detection)
    return filtered


@router.get("/health")
def detection_health():
    return {
        **road_model_health(),
        **urban_detection_health(),
        **anpr_model_health(),
    }


@router.post("/road")
async def detect_road(
    file: UploadFile = File(...),
    confidence: float = 0.12,
):
    backend_started = datetime.now(timezone.utc)
    upload_started = time.perf_counter()
    if confidence < 0.01 or confidence > 1.0:
        raise HTTPException(
            status_code=400,
            detail="Confidence must be between 0.01 and 1.0",
        )

    raw = await file.read(MAX_FILE_SIZE + 1)
    await file.close()
    upload_read_ms = round((time.perf_counter() - upload_started) * 1000, 2)

    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(raw) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Image exceeds 5 MB")

    try:
        detections, timing = await run_in_threadpool(
            detect_road_defects,
            raw,
            confidence,
        )
        detections = _prefer_model_over_water_heuristic(detections)
    except ValueError:
        raise HTTPException(status_code=422, detail="Upload a valid image")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception:
        logger.exception("Road AI inference failed")
        raise HTTPException(status_code=500, detail="Road AI inference failed")

    return {
        "model": road_model_health()["weight"],
        "detection_count": len(detections),
        "detections": detections,
        "timing": {
            **timing,
            "upload_read_ms": upload_read_ms,
            "backend_request_ms": round(
                (datetime.now(timezone.utc) - backend_started).total_seconds() * 1000,
                2,
            ),
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gps": None,
        "requires_manual_verification": True,
        "status": "CUSTOM TRAINED / FIELD VALIDATION REQUIRED",
    }


@router.post("/anpr")
async def detect_anpr(file: UploadFile = File(...)):
    """Prototype ANPR endpoint. Never fabricates a plate when OCR is uncertain."""
    backend_started = datetime.now(timezone.utc)
    upload_started = time.perf_counter()
    raw = await file.read(MAX_FILE_SIZE + 1)
    await file.close()
    upload_read_ms = round((time.perf_counter() - upload_started) * 1000, 2)
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(raw) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Image exceeds 5 MB")
    try:
        result = await run_in_threadpool(recognize_plate, raw)
        result["timing"]["upload_read_ms"] = upload_read_ms
        result["timing"]["backend_request_ms"] = round(
            (datetime.now(timezone.utc) - backend_started).total_seconds() * 1000,
            2,
        )
        return result
    except ValueError:
        raise HTTPException(status_code=422, detail="Upload a valid vehicle/plate image")
    except Exception:
        logger.exception("ANPR inference failed")
        raise HTTPException(status_code=500, detail="ANPR inference failed")


@router.post("/traffic")
async def detect_traffic(
    file: UploadFile = File(...),
    confidence: float = 0.25,
):
    if confidence < 0.01 or confidence > 1.0:
        raise HTTPException(status_code=400, detail="Confidence must be between 0.01 and 1.0")
    raw = await file.read(MAX_FILE_SIZE + 1)
    await file.close()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(raw) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Image exceeds 5 MB")
    try:
        result = await run_in_threadpool(detect_traffic_frame, raw, confidence)
    except ValueError:
        raise HTTPException(status_code=422, detail="Upload a valid traffic image")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception:
        logger.exception("Traffic AI inference failed")
        raise HTTPException(status_code=500, detail="Traffic AI inference failed")
    return {**result, "timestamp": datetime.now(timezone.utc).isoformat(), "gps": None}


@router.post("/infrastructure")
async def detect_infrastructure(file: UploadFile = File(...)):
    raw = await file.read(MAX_FILE_SIZE + 1)
    await file.close()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(raw) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Image exceeds 5 MB")
    try:
        result = await run_in_threadpool(analyze_infrastructure, raw)
    except ValueError:
        raise HTTPException(status_code=422, detail="Upload a valid road/infrastructure image")
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception:
        logger.exception("Infrastructure prototype inference failed")
        raise HTTPException(status_code=500, detail="Infrastructure prototype inference failed")
    return {**result, "timestamp": datetime.now(timezone.utc).isoformat(), "gps": None}


@router.post("/safety")
async def detect_safety(
    file: UploadFile = File(...),
    confidence: float = 0.25,
):
    if confidence < 0.01 or confidence > 1.0:
        raise HTTPException(status_code=400, detail="Confidence must be between 0.01 and 1.0")
    raw = await file.read(MAX_FILE_SIZE + 1)
    await file.close()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(raw) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Image exceeds 5 MB")
    try:
        result = await run_in_threadpool(analyze_safety, raw, confidence)
    except ValueError:
        raise HTTPException(status_code=422, detail="Upload a valid traffic/safety image")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception:
        logger.exception("Safety prototype inference failed")
        raise HTTPException(status_code=500, detail="Safety prototype inference failed")
    return {**result, "timestamp": datetime.now(timezone.utc).isoformat(), "gps": None}
