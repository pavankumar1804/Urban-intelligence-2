from pathlib import Path
from io import BytesIO
import os
import threading
import time

import numpy as np
from PIL import Image, ImageEnhance, ImageOps, UnidentifiedImageError

from .model_runtime import collect_released_memory, serialized_model_operation

try:
    import cv2
except ImportError:  # pragma: no cover - OpenCV is optional in some edge installs.
    cv2 = None

try:
    import torch
    from ultralytics import YOLO
except ImportError:  # Optional: edge nodes normally perform GPU inference.
    torch = None
    YOLO = None


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PT_MODEL_PATH = Path(__file__).resolve().parents[1] / "weights" / "road_defect_best.pt"
WATERLOGGING_MODEL_PATH = Path(__file__).resolve().parents[1] / "weights" / "waterlogging_best.pt"
ONNX_MODEL_PATH = PROJECT_ROOT / "frontend" / "ml" / "weights" / "road_defect_best.onnx"
PRETRAINED_POTHOLE_MODEL_PATHS = [
    PROJECT_ROOT / "frontend" / "ml" / "weights" / "candidates" / "vinothvikas1987_road_distress_yolov8_best.pt",
    PROJECT_ROOT / "frontend" / "ml" / "weights" / "candidates" / "peterhdd_pothole_yolov8_best.pt",
]
# Prefer the validated PyTorch checkpoint in cloud deployment. The current ONNX
# export is incompatible with Render's ONNX Runtime graph support (Split
# num_outputs), so ONNX remains opt-in until a compatible export is validated.
USE_ONNX = os.getenv("ROAD_AI_USE_ONNX", "0").strip().lower() in {"1", "true", "yes"}
MODEL_PATH = ONNX_MODEL_PATH if USE_ONNX and ONNX_MODEL_PATH.exists() else PT_MODEL_PATH

_model = None
_waterlogging_model = None
_pretrained_pothole_model = None
_pretrained_pothole_model_path = None
_inference_lock = threading.Lock()
_model_load_ms = None
_model_warmup_ms = None
# Benchmarking on RDD-style pothole images showed 640px materially improves
# pothole box recall versus 320px while keeping warm CPU inference practical.
INFERENCE_SIZE = int(os.getenv("ROAD_AI_IMGSZ", "640"))
PREPROCESS_MAX_SIDE = int(os.getenv("ROAD_AI_MAX_SIDE", "1024"))
ROAD_NMS_IOU = float(os.getenv("ROAD_AI_NMS_IOU", "0.70"))
ROAD_POTHOLE_MIN_CONFIDENCE = float(os.getenv("ROAD_AI_POTHOLE_MIN_CONF", "0.08"))
FUSION_IOU = float(os.getenv("ROAD_AI_FUSION_IOU", "0.35"))
USE_PRETRAINED_POTHOLE_MODEL = os.getenv("ROAD_AI_USE_PRETRAINED_POTHOLE", "0").strip().lower() in {"1", "true", "yes"}
USE_TILED_INFERENCE = os.getenv("ROAD_AI_TILED", "0").strip().lower() in {"1", "true", "yes"}
TILE_TRIGGER_SIDE = int(os.getenv("ROAD_AI_TILE_TRIGGER_SIDE", "900"))
TILE_SIZE = int(os.getenv("ROAD_AI_TILE_SIZE", "512"))
TILE_INFERENCE_SIZE = int(os.getenv("ROAD_AI_TILE_IMGSZ", "416"))
TILE_OVERLAP = float(os.getenv("ROAD_AI_TILE_OVERLAP", "0.12"))
MAX_TILES = int(os.getenv("ROAD_AI_MAX_TILES", "6"))


def get_model():
    global _model, _model_load_ms, MODEL_PATH, _waterlogging_model

    if YOLO is None:
        raise RuntimeError(
            "Backend AI detection is not installed. Use the edge inference "
            "pipeline, or install the optional ultralytics dependency."
        )

    if _model is None:
        # The dedicated waterlogging checkpoint is used sequentially later.
        # Release it before loading the primary road checkpoint.
        if _waterlogging_model is not None:
            _waterlogging_model = None
            collect_released_memory()
        if torch is not None:
            torch.set_num_threads(max(1, int(os.getenv("TORCH_NUM_THREADS", "1"))))
        if not MODEL_PATH.exists() and not PT_MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

        load_started = time.perf_counter()
        try:
            _model = YOLO(str(MODEL_PATH))
        except Exception:
            if PT_MODEL_PATH.exists() and MODEL_PATH != PT_MODEL_PATH:
                MODEL_PATH = PT_MODEL_PATH
                _model = YOLO(str(PT_MODEL_PATH))
            else:
                raise
        _model_load_ms = round((time.perf_counter() - load_started) * 1000, 2)

    return _model


def get_waterlogging_model():
    """Load the dedicated trained waterlogging segmentation model."""
    global _waterlogging_model, _model, _pretrained_pothole_model

    if YOLO is None:
        return None

    if not WATERLOGGING_MODEL_PATH.exists():
        return None

    if _waterlogging_model is None:
        # Primary/pothole inference is complete before this function runs.
        # Keep only one YOLO checkpoint resident at a time.
        _model = None
        _pretrained_pothole_model = None
        collect_released_memory()
        _waterlogging_model = YOLO(str(WATERLOGGING_MODEL_PATH))

    return _waterlogging_model


def get_pretrained_pothole_model():
    global _pretrained_pothole_model, _pretrained_pothole_model_path
    if YOLO is None:
        return None
    if _pretrained_pothole_model is not None:
        return _pretrained_pothole_model
    for candidate in PRETRAINED_POTHOLE_MODEL_PATHS:
        if candidate.exists():
            _pretrained_pothole_model_path = candidate
            _pretrained_pothole_model = YOLO(str(candidate))
            return _pretrained_pothole_model
    return None


def warm_road_model():
    """Initialize the inference backend and run one synthetic frame at startup."""
    global _model_warmup_ms, MODEL_PATH, _model
    model = get_model()
    if _model_warmup_ms is None:
        started = time.perf_counter()
        frame = np.zeros((INFERENCE_SIZE, INFERENCE_SIZE, 3), dtype=np.uint8)
        try:
            with _inference_lock:
                model.predict(
                    source=frame,
                    conf=0.18,
                    verbose=False,
                    imgsz=INFERENCE_SIZE,
                    device="cpu",
                )
        except Exception:
            if PT_MODEL_PATH.exists() and MODEL_PATH != PT_MODEL_PATH:
                MODEL_PATH = PT_MODEL_PATH
                _model = YOLO(str(PT_MODEL_PATH))
                with _inference_lock:
                    _model.predict(
                        source=frame,
                        conf=0.18,
                        verbose=False,
                        imgsz=INFERENCE_SIZE,
                        device="cpu",
                    )
                model = _model
            else:
                raise
        _model_warmup_ms = round((time.perf_counter() - started) * 1000, 2)
    return model


def release_road_models() -> None:
    global _model, _waterlogging_model, _pretrained_pothole_model
    _model = None
    _waterlogging_model = None
    _pretrained_pothole_model = None
    collect_released_memory()


def _collect_waterlogging_detections(
    frame,
    scale_x,
    scale_y,
    confidence=0.50,
):
    """Run the dedicated waterlogging model and return API-compatible boxes."""
    model = get_waterlogging_model()

    if model is None:
        return []

    with _inference_lock:
        results = model.predict(
            source=frame,
            conf=confidence,
            verbose=False,
            imgsz=640,
            iou=0.70,
            max_det=100,
            device="cpu",
        )

    detections = []

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            score = float(box.conf[0].item())

            if score < confidence:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "class_id": 0,
                "class_name": "waterlogging",
                "confidence": score,
                "bbox": {
                    "x1": round(x1 * scale_x, 2),
                    "y1": round(y1 * scale_y, 2),
                    "x2": round(x2 * scale_x, 2),
                    "y2": round(y2 * scale_y, 2),
                },
                "detection_method": "CUSTOM YOLO / WATERLOGGING SEGMENTATION",
            })

    return detections


def _collect_road_model_detections(frame, scale_x, scale_y, confidence, imgsz=None):
    global _model, MODEL_PATH
    model = get_model()
    inference_confidence = confidence
    predict_size = int(imgsz or INFERENCE_SIZE)
    try:
        with _inference_lock:
            results = model.predict(
                source=frame,
                conf=inference_confidence,
                verbose=False,
                imgsz=predict_size,
                iou=ROAD_NMS_IOU,
                max_det=200,
                device="cpu",
            )
    except Exception:
        if PT_MODEL_PATH.exists() and MODEL_PATH != PT_MODEL_PATH:
            MODEL_PATH = PT_MODEL_PATH
            _model = YOLO(str(PT_MODEL_PATH))
            with _inference_lock:
                results = _model.predict(
                    source=frame,
                    conf=inference_confidence,
                    verbose=False,
                    imgsz=predict_size,
                    iou=ROAD_NMS_IOU,
                    max_det=200,
                    device="cpu",
                )
        else:
            raise

    detections = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            class_name = str(result.names.get(class_id, "")).strip()
            score = float(box.conf[0].item())
            if score < confidence:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": round(score, 4),
                    "raw_model_confidence": round(score, 4),
                    "detection_method": "CUSTOM YOLO / TRAINED MODEL",
                    "bbox": {
                        "x1": round(float(x1) * scale_x, 2),
                        "y1": round(float(y1) * scale_y, 2),
                        "x2": round(float(x2) * scale_x, 2),
                        "y2": round(float(y2) * scale_y, 2),
                    },
                }
            )
    return detections


def _collect_pothole_detections(frame, scale_x, scale_y, confidence):
    if not USE_PRETRAINED_POTHOLE_MODEL:
        return []
    candidate_model = get_pretrained_pothole_model()
    if candidate_model is None:
        return []
    try:
        with _inference_lock:
            results = candidate_model.predict(
                source=frame,
                conf=confidence,
                verbose=False,
                imgsz=INFERENCE_SIZE,
                device="cpu",
            )
    except Exception:
        return []

    detections = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            name = str(result.names.get(class_id, "")).strip()
            if "pothole" not in name.lower():
                continue
            score = float(box.conf[0].item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": "pothole",
                    "confidence": round(score, 4),
                    "bbox": {
                        "x1": round(float(x1) * scale_x, 2),
                        "y1": round(float(y1) * scale_y, 2),
                        "x2": round(float(x2) * scale_x, 2),
                        "y2": round(float(y2) * scale_y, 2),
                    },
                    "detection_method": "PRETRAINED-POTHOLE MODEL",
                }
            )
    return _nms_by_class(detections, ROAD_NMS_IOU)


def _collect_tiled_detections(image_rgb, original_width, original_height, confidence):
    """Sequential tiled inference for small/multiple defects without RAM spikes."""
    if not USE_TILED_INFERENCE:
        return []
    width, height = image_rgb.size
    if max(width, height) < TILE_TRIGGER_SIDE:
        return []

    tile = max(256, min(TILE_SIZE, width, height))
    overlap = max(0.0, min(0.40, TILE_OVERLAP))
    step = max(64, int(tile * (1.0 - overlap)))
    xs = list(range(0, max(1, width - tile + 1), step))
    ys = list(range(0, max(1, height - tile + 1), step))
    if not xs or xs[-1] != max(0, width - tile):
        xs.append(max(0, width - tile))
    if not ys or ys[-1] != max(0, height - tile):
        ys.append(max(0, height - tile))

    positions = [(x, y) for y in ys for x in xs]
    if len(positions) > MAX_TILES:
        indexes = np.linspace(0, len(positions) - 1, MAX_TILES, dtype=int)
        positions = [positions[int(i)] for i in indexes]

    detections = []
    for x, y in positions:
        crop = image_rgb.crop((x, y, x + tile, y + tile))
        frame = np.asarray(crop).copy()
        crop.close()
        local = _collect_road_model_detections(frame, 1.0, 1.0, confidence, TILE_INFERENCE_SIZE)
        for det in local:
            box = det["bbox"]
            box["x1"] = round((box["x1"] + x) * original_width / width, 2)
            box["x2"] = round((box["x2"] + x) * original_width / width, 2)
            box["y1"] = round((box["y1"] + y) * original_height / height, 2)
            box["y2"] = round((box["y2"] + y) * original_height / height, 2)
            det["detection_method"] = det.get("detection_method", "CUSTOM YOLO") + " / TILED"
            detections.append(det)
        del frame, local

    return _nms_by_class(detections, ROAD_NMS_IOU)


@serialized_model_operation
def detect_road_defects(raw: bytes, confidence: float = 0.12):
    global MODEL_PATH, _model
    from .anpr_service import release_anpr_model
    from .urban_vision import release_traffic_model

    release_anpr_model()
    release_traffic_model()
    request_started = time.perf_counter()
    decode_started = request_started
    try:
        with Image.open(BytesIO(raw)) as source:
            source.load()
            image = ImageOps.exif_transpose(source).convert("RGB")
            original_width, original_height = image.size
            # Render runs with tiled inference disabled. Avoid retaining a
            # second full decoded image unless tiled inference is explicitly
            # enabled for a deployment with enough memory.
            tile_image = image.copy() if USE_TILED_INFERENCE else None
            decoded_image = image
    except (UnidentifiedImageError, OSError):
        raise ValueError("Invalid image")

    decode_ms = (time.perf_counter() - decode_started) * 1000
    preprocess_started = time.perf_counter()
    decoded_image.thumbnail((INFERENCE_SIZE, INFERENCE_SIZE))
    if tile_image is not None:
        tile_image.thumbnail((PREPROCESS_MAX_SIDE, PREPROCESS_MAX_SIDE))
    inference_width, inference_height = decoded_image.size
    scale_x = original_width / inference_width
    scale_y = original_height / inference_height
    frame = np.asarray(decoded_image).copy()
    decoded_image.close()
    preprocess_ms = (time.perf_counter() - preprocess_started) * 1000

    model_started = time.perf_counter()
    get_model()
    model_ready_ms = (time.perf_counter() - model_started) * 1000

    inference_started = time.perf_counter()
    detections = _collect_road_model_detections(frame, scale_x, scale_y, confidence)

    # If the first pass is weak, a lightly contrast-enhanced pass can recover
    # pothole edges/shadows without changing or fabricating model confidence.
    strongest = max((float(d["confidence"]) for d in detections), default=0.0)
    if strongest < 0.45:
        enhanced_image = Image.fromarray(frame)
        enhanced_image = ImageEnhance.Contrast(enhanced_image).enhance(1.18)
        enhanced_frame = np.asarray(enhanced_image).copy()
        enhanced_image.close()
        enhanced = _collect_road_model_detections(
            enhanced_frame, scale_x, scale_y, confidence, INFERENCE_SIZE
        )
        for item in enhanced:
            item["detection_method"] = item.get("detection_method", "CUSTOM YOLO") + " / CONTRAST"
        detections.extend(enhanced)
        del enhanced_frame, enhanced

    tiled_detections = _collect_tiled_detections(
        tile_image, original_width, original_height, confidence
    ) if tile_image is not None else []
    if tile_image is not None:
        tile_image.close()
    if tiled_detections:
        detections.extend(tiled_detections)

    pothole_detections = _collect_pothole_detections(
        frame, scale_x, scale_y, confidence
    )
    if pothole_detections:
        detections.extend(pothole_detections)
    detections = _fuse_same_class_evidence(detections)
    detections = _nms_by_class(detections, ROAD_NMS_IOU)
    inference_ms = (time.perf_counter() - inference_started) * 1000

    postprocess_started = time.perf_counter()
    waterlogging_started = time.perf_counter()

    # Run the dedicated trained waterlogging model first.
    trained_waterlogging = _collect_waterlogging_detections(
        frame,
        scale_x,
        scale_y,
        confidence=0.50,
    )
    if trained_waterlogging:
        detections.extend(trained_waterlogging)

    # Keep the prototype water detector available for regions the trained
    # model did not classify. The API applies spatial/model-priority filtering
    # so a heuristic result cannot replace a trained pothole or waterlogging.
    heuristic_detections = _detect_waterlogging(frame, scale_x, scale_y)
    model_waterlogging = [
        d for d in detections
        if d["class_name"].lower() == "waterlogging"
        and d.get("detection_method") != "COMPUTER-VISION PROTOTYPE"
    ]
    # Preserve trained waterlogging predictions even if the prototype's large
    # region would otherwise win class NMS by raw score.
    heuristic_detections = [
        heuristic for heuristic in heuristic_detections
        if not any(
            _box_iou(heuristic, model_water) >= 0.10
            or _intersection_over_source(heuristic["bbox"], model_water["bbox"]) >= 0.35
            for model_water in model_waterlogging
        )
    ]
    detections.extend(heuristic_detections)
    detections = _nms_by_class(detections, ROAD_NMS_IOU)
    waterlogging_ms = (time.perf_counter() - waterlogging_started) * 1000

    del frame
    if torch is not None:
        import gc
        gc.collect()

    postprocess_ms = (time.perf_counter() - postprocess_started) * 1000
    total_ms = (time.perf_counter() - request_started) * 1000
    timing = {
        "image_decode_ms": round(decode_ms, 2),
        "preprocess_ms": round(preprocess_ms, 2),
        "model_load_ms": round(model_ready_ms, 2),
        "model_ready_ms": round(model_ready_ms, 2),
        "inference_ms": round(inference_ms, 2),
        "waterlogging_ms": round(waterlogging_ms, 2),
        "postprocess_ms": round(postprocess_ms, 2),
        "total_ms": round(total_ms, 2),
    }
    return detections, timing


def _box_iou(a, b):
    ax1, ay1, ax2, ay2 = a["bbox"]["x1"], a["bbox"]["y1"], a["bbox"]["x2"], a["bbox"]["y2"]
    bx1, by1, bx2, by2 = b["bbox"]["x1"], b["bbox"]["y1"], b["bbox"]["x2"], b["bbox"]["y2"]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / max(1e-6, area_a + area_b - inter)


def _intersection_over_source(source, other):
    sx1, sy1, sx2, sy2 = source["x1"], source["y1"], source["x2"], source["y2"]
    ox1, oy1, ox2, oy2 = other["x1"], other["y1"], other["x2"], other["y2"]
    intersection = max(0, min(sx2, ox2) - max(sx1, ox1)) * max(0, min(sy2, oy2) - max(sy1, oy1))
    source_area = max(0, sx2 - sx1) * max(0, sy2 - sy1)
    return intersection / max(1e-6, source_area)


def _fuse_same_class_evidence(detections):
    """Fuse corroborating full-frame/tile detections without inventing confidence."""
    if not detections:
        return detections
    output = []
    used = set()
    for i, base in enumerate(detections):
        if i in used:
            continue
        group = [base]
        used.add(i)
        for j in range(i + 1, len(detections)):
            other = detections[j]
            if j in used or other["class_name"] != base["class_name"]:
                continue
            if _box_iou(base, other) >= FUSION_IOU:
                group.append(other)
                used.add(j)
        best = max(group, key=lambda d: d["confidence"]).copy()
        if len(group) > 1:
            # Multiple passes corroborate one box, but confidence remains the
            # highest actual model score. Never turn low model confidence into
            # an inflated percentage merely because passes agreed.
            best["confidence"] = round(float(best["confidence"]), 4)
            best["raw_model_confidence"] = round(
                max(float(item.get("raw_model_confidence", item["confidence"])) for item in group), 4
            )
            best["evidence_count"] = len(group)
            best["confidence_method"] = "MULTI-PASS EVIDENCE FUSION"
        output.append(best)
    return output


def _nms_by_class(detections, threshold):
    kept = []
    for class_name in sorted({d["class_name"] for d in detections}):
        class_dets = sorted(
            [d for d in detections if d["class_name"] == class_name],
            key=lambda item: item["confidence"],
            reverse=True,
        )
        while class_dets:
            best = class_dets.pop(0)
            kept.append(best)
            class_dets = [d for d in class_dets if _box_iou(best, d) < threshold]
    return sorted(kept, key=lambda item: item["confidence"], reverse=True)


def _detect_waterlogging(frame_rgb, scale_x, scale_y):
    """Conservative standing-water heuristic; avoid vehicles/crash debris."""
    if cv2 is None:
        return []
    height, width = frame_rgb.shape[:2]
    if height < 40 or width < 40:
        return []

    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Standing water should occupy the lower road plane, not the middle/top of
    # the scene where vehicles, barriers and crash debris commonly appear.
    lower_y = int(height * 0.55)
    roi = hsv[lower_y:, :]
    blue_gray_water = cv2.inRange(roi, (80, 8, 45), (135, 95, 235))
    muddy_water = cv2.inRange(roi, (7, 20, 40), (38, 150, 220))
    mask = blue_gray_water | muddy_water
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    roi_area = max(1, roi.shape[0] * roi.shape[1])
    for contour in contours:
        area = cv2.contourArea(contour)
        area_ratio = area / roi_area
        if area_ratio < 0.18 or area_ratio > 0.70:
            continue
        x, y0, w, h = cv2.boundingRect(contour)
        if w < width * 0.28 or h < 12:
            continue
        aspect = w / max(1, h)
        # Water accumulation normally spreads horizontally across the road.
        if aspect < 2.0:
            continue
        y = y0 + lower_y
        center_y = y + h / 2
        if center_y < height * 0.68:
            continue

        crop_hsv = hsv[y:y+h, x:x+w]
        crop_bgr = bgr[y:y+h, x:x+w]
        if not crop_hsv.size or not crop_bgr.size:
            continue
        saturation = float(np.median(crop_hsv[:, :, 1]))
        # Crash scenes/vehicles have dense edges and texture; standing water is
        # generally smoother. Reject highly textured candidate regions.
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        edge_density = float(np.count_nonzero(cv2.Canny(gray, 70, 150))) / max(1, gray.size)
        if saturation > 135 or edge_density > 0.12:
            continue

        score = min(0.70, 0.30 + area_ratio * 0.75 + max(0.0, 0.12 - edge_density))
        if score < 0.50:
            continue
        detections.append({
            "class_id": 5,
            "class_name": "waterlogging",
            "confidence": round(float(score), 4),
            "bbox": {
                "x1": round(float(x) * scale_x, 2),
                "y1": round(float(y) * scale_y, 2),
                "x2": round(float(x + w) * scale_x, 2),
                "y2": round(float(y + h) * scale_y, 2),
            },
            "detection_method": "COMPUTER-VISION PROTOTYPE",
            "requires_manual_verification": True,
        })
    return _nms_by_class(detections, 0.30)

def road_model_health():
    """Lightweight readiness check that does not load the model into memory."""
    return {
        "road_model_ready": MODEL_PATH.exists(),
        "weight": MODEL_PATH.name,
        "ultralytics_ready": YOLO is not None,
        "engine": "onnxruntime" if MODEL_PATH.suffix == ".onnx" else "pytorch",
        "inference_size": INFERENCE_SIZE,
        "preprocess_max_side": PREPROCESS_MAX_SIDE,
        "device": "cpu",
        "model_load_ms": _model_load_ms,
        "model_warmup_ms": _model_warmup_ms,
    }
