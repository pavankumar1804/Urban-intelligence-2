from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from .model_runtime import collect_released_memory, serialized_model_operation

try:
    import cv2
except ImportError:  # pragma: no cover - OpenCV is optional in lighter deploys.
    cv2 = None

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - optional ML runtime.
    YOLO = None


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRAFFIC_MODEL_PATH = PROJECT_ROOT / "frontend" / "ml" / "weights" / "traffic_coco.pt"
TRAFFIC_RUNTIME_FALLBACK = "yolov8n.pt"
TRAFFIC_CLASSES = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}

_traffic_model = None
_traffic_model_name = None
_traffic_lock = threading.Lock()


@dataclass
class DecodedImage:
    frame: np.ndarray
    original_width: int
    original_height: int
    inference_width: int
    inference_height: int


def _decode_image(raw: bytes, max_side: int = 960) -> DecodedImage:
    try:
        with Image.open(BytesIO(raw)) as image:
            image.load()
            image = image.convert("RGB")
            original_width, original_height = image.size
            image.thumbnail((max_side, max_side))
            inference_width, inference_height = image.size
            frame = np.array(image)
    except (UnidentifiedImageError, OSError):
        raise ValueError("Invalid image")

    return DecodedImage(
        frame=frame,
        original_width=original_width,
        original_height=original_height,
        inference_width=inference_width,
        inference_height=inference_height,
    )


def _scale_box(decoded: DecodedImage, xyxy: list[float]) -> dict[str, float]:
    scale_x = decoded.original_width / decoded.inference_width
    scale_y = decoded.original_height / decoded.inference_height
    x1, y1, x2, y2 = xyxy
    return {
        "x1": round(float(x1) * scale_x, 2),
        "y1": round(float(y1) * scale_y, 2),
        "x2": round(float(x2) * scale_x, 2),
        "y2": round(float(y2) * scale_y, 2),
    }


def _traffic_status() -> dict[str, Any]:
    return {
        "model_ready": TRAFFIC_MODEL_PATH.exists() or YOLO is not None,
        "weight": TRAFFIC_MODEL_PATH.name if TRAFFIC_MODEL_PATH.exists() else TRAFFIC_RUNTIME_FALLBACK,
        "engine": "pytorch",
        "method": "PRETRAINED COCO YOLO",
        "supported_classes": sorted(TRAFFIC_CLASSES),
        "unsupported_classes": ["auto_rickshaw", "emergency_vehicle"],
        "runtime_fallback": None if TRAFFIC_MODEL_PATH.exists() else "Ultralytics COCO nano weights",
    }


def _get_traffic_model():
    global _traffic_model, _traffic_model_name
    if YOLO is None:
        raise RuntimeError("Ultralytics is not installed")
    if _traffic_model is None:
        if TRAFFIC_MODEL_PATH.exists():
            _traffic_model_name = TRAFFIC_MODEL_PATH.name
            _traffic_model = YOLO(str(TRAFFIC_MODEL_PATH))
        else:
            _traffic_model_name = TRAFFIC_RUNTIME_FALLBACK
            _traffic_model = YOLO(TRAFFIC_RUNTIME_FALLBACK)
    return _traffic_model


def release_traffic_model() -> None:
    global _traffic_model
    _traffic_model = None
    collect_released_memory()


def detection_health() -> dict[str, Any]:
    return {
        "traffic": _traffic_status(),
        "infrastructure": {
            "method": "COMPUTER-VISION PROTOTYPE",
            "trained_model_ready": False,
            "outputs": [
                "waterlogging",
                "possible_zebra_crossing",
                "possible_traffic_sign",
                "possible_divider",
                "possible_missing_infrastructure",
            ],
        },
    }


@serialized_model_operation
def detect_traffic(raw: bytes, confidence: float = 0.25) -> dict[str, Any]:
    from .anpr_service import release_anpr_model
    from .road_detector import release_road_models

    release_anpr_model()
    release_road_models()
    started = time.perf_counter()
    decoded = _decode_image(raw)
    model = _get_traffic_model()

    with _traffic_lock:
        results = model.predict(
            source=decoded.frame,
            conf=confidence,
            imgsz=640,
            device="cpu",
            verbose=False,
        )

    detections = []
    counts = {name: 0 for name in sorted(TRAFFIC_CLASSES)}
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            class_name = str(result.names[class_id])
            if class_name not in TRAFFIC_CLASSES:
                continue
            counts[class_name] += 1
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": round(float(box.conf[0].item()), 4),
                    "bbox": _scale_box(decoded, box.xyxy[0].tolist()),
                    "tracking_status": "single_frame_no_track_id",
                }
            )

    vehicle_count = sum(counts[name] for name in ("bicycle", "car", "motorcycle", "bus", "truck"))
    person_count = counts["person"]
    density_score = min(1.0, vehicle_count / 12)
    if density_score >= 0.67:
        density_level = "HIGH"
    elif density_score >= 0.34:
        density_level = "MEDIUM"
    else:
        density_level = "LOW"

    return {
        "model": _traffic_model_name or (TRAFFIC_MODEL_PATH.name if TRAFFIC_MODEL_PATH.exists() else TRAFFIC_RUNTIME_FALLBACK),
        "method": "PRETRAINED",
        "source": "Ultralytics YOLO COCO pretrained weights",
        "license": "AGPL-3.0 software; COCO pretrained classes",
        "detection_count": len(detections),
        "detections": detections,
        "counts_by_class": counts,
        "vehicle_count": vehicle_count,
        "pedestrian_count": person_count,
        "density_score": round(density_score, 2),
        "density_level": density_level,
        "bottleneck": vehicle_count >= 9,
        "unsupported": {
            "auto_rickshaw": "not reliably available in COCO weights",
            "emergency_vehicle": "not reliably available in COCO weights",
        },
        "requires_manual_verification": True,
        "timing": {"total_ms": round((time.perf_counter() - started) * 1000, 2)},
    }


def analyze_infrastructure(raw: bytes) -> dict[str, Any]:
    started = time.perf_counter()
    if cv2 is None:
        raise RuntimeError("OpenCV is not installed")
    decoded = _decode_image(raw)
    rgb = decoded.frame
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    height, width = rgb.shape[:2]
    lower_half = bgr[height // 2 :, :]
    hsv = cv2.cvtColor(lower_half, cv2.COLOR_BGR2HSV)

    water_mask = cv2.inRange(hsv, (85, 15, 25), (130, 120, 230))
    water_ratio = float(np.count_nonzero(water_mask)) / max(1, water_mask.size)
    waterlogging = water_ratio > 0.18

    gray = cv2.cvtColor(lower_half, cv2.COLOR_BGR2GRAY)
    _, bright = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stripe_like = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area > 80 and w > h * 1.8:
            stripe_like += 1
    possible_zebra = stripe_like >= 10

    red1 = cv2.inRange(hsv, (0, 80, 70), (12, 255, 255))
    red2 = cv2.inRange(hsv, (168, 80, 70), (180, 255, 255))
    blue = cv2.inRange(hsv, (95, 80, 60), (125, 255, 255))
    sign_ratio = float(np.count_nonzero(red1 | red2 | blue)) / max(1, red1.size)
    possible_sign = sign_ratio > 0.08

    edges = cv2.Canny(gray, 80, 160)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=height // 5, maxLineGap=20)
    divider_lines = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = np.asarray(line).reshape(-1)[:4]
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            if dy > dx * 1.5 and width * 0.3 <= (x1 + x2) / 2 <= width * 0.7:
                divider_lines += 1
    possible_divider = divider_lines > 0

    findings = [
        {
            "event_type": "waterlogging",
            "detected": waterlogging,
            "confidence": round(min(0.75, water_ratio * 3), 2) if waterlogging else 0.0,
            "method": "COMPUTER-VISION PROTOTYPE",
            "requires_manual_verification": True,
        },
        {
            "event_type": "zebra_crossing",
            "detected": possible_zebra,
            "confidence": round(min(0.7, stripe_like / 10), 2) if possible_zebra else 0.0,
            "method": "COMPUTER-VISION PROTOTYPE",
            "requires_manual_verification": True,
        },
        {
            "event_type": "traffic_sign",
            "detected": possible_sign,
            "confidence": round(min(0.65, sign_ratio * 8), 2) if possible_sign else 0.0,
            "method": "COMPUTER-VISION PROTOTYPE",
            "requires_manual_verification": True,
        },
        {
            "event_type": "road_divider",
            "detected": possible_divider,
            "confidence": 0.45 if possible_divider else 0.0,
            "method": "COMPUTER-VISION PROTOTYPE",
            "requires_manual_verification": True,
        },
    ]

    missing = []
    if not possible_zebra:
        missing.append({"event_type": "possible_missing_zebra", "possible_missing": True, "verification_required": True})
    if not possible_divider:
        missing.append({"event_type": "possible_missing_divider", "possible_missing": True, "verification_required": True})
    if not possible_sign:
        missing.append({"event_type": "possible_missing_sign", "possible_missing": True, "verification_required": True})

    return {
        "method": "COMPUTER-VISION PROTOTYPE",
        "status": "PROTOTYPE / REQUIRES VALIDATION",
        "findings": findings,
        "missing_infrastructure": missing,
        "timing": {"total_ms": round((time.perf_counter() - started) * 1000, 2)},
    }


def analyze_safety(raw: bytes, confidence: float = 0.25) -> dict[str, Any]:
    traffic = detect_traffic(raw, confidence)
    vehicles = [d for d in traffic["detections"] if d["class_name"] != "person"]
    people = [d for d in traffic["detections"] if d["class_name"] == "person"]
    risks = []

    def center(det: dict[str, Any]) -> tuple[float, float]:
        box = det["bbox"]
        return ((box["x1"] + box["x2"]) / 2, (box["y1"] + box["y2"]) / 2)

    for person in people:
        px, py = center(person)
        nearest = None
        for vehicle in vehicles:
            vx, vy = center(vehicle)
            distance = ((px - vx) ** 2 + (py - vy) ** 2) ** 0.5
            if nearest is None or distance < nearest[0]:
                nearest = (distance, vehicle)
        if nearest and nearest[0] < max(80, decoded_distance_threshold(person)):
            risks.append(
                {
                    "event_type": "pedestrian_risk",
                    "confidence": round(min(person["confidence"], nearest[1]["confidence"]), 2),
                    "person_bbox": person["bbox"],
                    "vehicle_bbox": nearest[1]["bbox"],
                    "vehicle_class": nearest[1]["class_name"],
                    "method": "RULE-BASED",
                    "requires_manual_verification": True,
                }
            )

    return {
        "method": "RULE-BASED",
        "traffic_method": traffic["method"],
        "pedestrian_risks": risks,
        "rash_driving": {
            "status": "PROTOTYPE",
            "method": "RULE-BASED tracking over video is required",
            "detected": False,
            "requires_manual_verification": True,
        },
        "hit_and_run": {
            "status": "PROTOTYPE",
            "method": "RULE-BASED tracking plus collision/departure evidence is required",
            "detected": False,
            "requires_manual_verification": True,
        },
    }


def decoded_distance_threshold(person: dict[str, Any]) -> float:
    box = person["bbox"]
    return max(box["x2"] - box["x1"], box["y2"] - box["y1"]) * 1.5
