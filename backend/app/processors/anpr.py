"""
Urban Intelligence Platform - Edge ANPR Processor

Automatic Number Plate Recognition (ANPR) pipeline for incident-associated vehicles.
Includes plate localization, image enhancement, OCR extraction, Indian registration
format syntax validation, and confidence calibration.
"""
import re
import os
import shutil
import uuid
import time
from collections import Counter
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger("uvicorn.error")


class ONNXPlateDetector:
    """Small YOLO detection wrapper that avoids importing PyTorch at runtime."""

    def __init__(self, model_path: Path, input_size: int):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, int(os.getenv("TORCH_NUM_THREADS", "1")))
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = input_size

    def detect(self, frame, confidence: float = 0.20) -> List[Dict[str, Any]]:
        import cv2
        import numpy as np

        height, width = frame.shape[:2]
        scale = min(self.input_size / width, self.input_size / height)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        pad_x = (self.input_size - resized_width) // 2
        pad_y = (self.input_size - resized_height) // 2
        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        canvas[pad_y:pad_y + resized_height, pad_x:pad_x + resized_width] = resized
        tensor = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        prediction = np.squeeze(self.session.run(None, {self.input_name: tensor[None]})[0])
        if prediction.ndim != 2:
            return []
        # YOLO11 detect exports [channels, anchors]; transpose to one row/anchor.
        if prediction.shape[0] <= 84 and prediction.shape[0] < prediction.shape[1]:
            prediction = prediction.T
        if prediction.shape[1] < 5:
            return []

        scores = prediction[:, 4:].max(axis=1)
        selected = np.flatnonzero(scores >= confidence)
        if not selected.size:
            return []

        nms_boxes = []
        selected_scores = []
        selected_xyxy = []
        for index in selected:
            center_x, center_y, box_width, box_height = prediction[index, :4]
            x1 = float(center_x - box_width / 2)
            y1 = float(center_y - box_height / 2)
            nms_boxes.append([int(x1), int(y1), int(box_width), int(box_height)])
            selected_scores.append(float(scores[index]))
            selected_xyxy.append([x1, y1, float(center_x + box_width / 2), float(center_y + box_height / 2)])

        kept = cv2.dnn.NMSBoxes(nms_boxes, selected_scores, confidence, 0.45)
        detections = []
        for raw_index in np.asarray(kept).reshape(-1):
            x1, y1, x2, y2 = selected_xyxy[int(raw_index)]
            detections.append({
                "confidence": selected_scores[int(raw_index)],
                "xyxy": [
                    max(0.0, min(float(width), (x1 - pad_x) / scale)),
                    max(0.0, min(float(height), (y1 - pad_y) / scale)),
                    max(0.0, min(float(width), (x2 - pad_x) / scale)),
                    max(0.0, min(float(height), (y2 - pad_y) / scale)),
                ],
            })
        return sorted(detections, key=lambda item: item["confidence"], reverse=True)


def _configure_tesseract(pytesseract) -> bool:
    """Configure common Windows installs when tesseract.exe is absent from PATH."""
    candidates = [
        shutil.which("tesseract"),
        Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Tesseract-OCR" / "tesseract.exe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Tesseract-OCR" / "tesseract.exe",
    ]
    executable = next((Path(path) for path in candidates if path and Path(path).is_file()), None)
    if executable:
        pytesseract.pytesseract.tesseract_cmd = str(executable)
        return True
    return False


@dataclass
class PlateResult:
    """Represents the output of the ANPR pipeline."""
    plate_id: str
    vehicle_track_id: int
    plate_number: str  # Validated registration or "UNKNOWN"
    raw_ocr_text: str
    plate_detection_confidence: float
    ocr_confidence: float
    overall_confidence: float
    is_format_valid: bool
    is_low_confidence: bool
    requires_manual_verification: bool
    explainability: List[str]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    plate_bbox: Optional[List[float]] = None
    localizer_status: str = "not_run"
    timings: Dict[str, float] = field(default_factory=dict)


class ANPRProcessor:
    """
    Edge ANPR Processor for localized plate recognition.
    Strictly avoids fabricating or hallucinating numbers when OCR is low-confidence.
    """

    # Standard Indian Motor Vehicle Registration Format (e.g. TS09AB1234, DL01C5678, MH12DE1423, KA03MJ9988)
    # State (2 letters) + RTO Code (2 digits) + Optional Series (1-3 letters) + Registration Number (4 digits)
    INDIAN_PLATE_REGEX = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{4}$")

    # Common OCR character confusion maps for automotive number plates
    CHAR_CORRECTIONS = {
        "O": "0", "I": "1", "Z": "2", "S": "5", "B": "8", "G": "6", "Q": "0"
    }
    NUM_CORRECTIONS = {
        "0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G"
    }

    @classmethod
    def _select_ocr_candidate(cls, candidates):
        """Prefer repeatable, format-valid OCR without altering its confidence."""
        prepared = []
        for text, confidence in candidates:
            normalized = re.sub(r"[^A-Z0-9]", "", str(text).upper())
            if normalized:
                prepared.append((normalized, max(0.0, min(1.0, float(confidence)))))
        if not prepared:
            return None

        observations = Counter(text for text, _ in prepared)
        return max(
            prepared,
            key=lambda item: (
                bool(cls.INDIAN_PLATE_REGEX.fullmatch(item[0])),
                observations[item[0]],
                6 <= len(item[0]) <= 13,
                item[1],
                len(item[0]),
            ),
        )

    @staticmethod
    def _rectify_plate_crop(plate_crop, cv2, np):
        """Perspective-correct a clearly bounded, skewed plate when possible."""
        if plate_crop is None or plate_crop.size == 0:
            return plate_crop
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 40, 130)
        edges = cv2.morphologyEx(
            edges, cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8), iterations=2
        )
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        crop_area = float(plate_crop.shape[0] * plate_crop.shape[1])

        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:12]:
            area_ratio = cv2.contourArea(contour) / crop_area if crop_area else 0.0
            if not 0.20 <= area_ratio <= 0.90:
                continue
            perimeter = cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(contour, 0.05 * perimeter, True)
            if len(polygon) != 4:
                continue

            points = polygon.reshape(4, 2).astype("float32")
            point_sum = points.sum(axis=1)
            point_diff = np.diff(points, axis=1).reshape(-1)
            ordered = np.array(
                [
                    points[np.argmin(point_sum)],
                    points[np.argmin(point_diff)],
                    points[np.argmax(point_sum)],
                    points[np.argmax(point_diff)],
                ],
                dtype="float32",
            )
            top_left, top_right, bottom_right, bottom_left = ordered
            width = int(max(np.linalg.norm(bottom_right - bottom_left), np.linalg.norm(top_right - top_left)))
            height = int(max(np.linalg.norm(top_right - bottom_right), np.linalg.norm(top_left - bottom_left)))
            if width < 80 or height < 25:
                continue
            if height > width:
                width, height = height, width
            aspect_ratio = width / max(height, 1)
            if not 1.4 <= aspect_ratio <= 6.0:
                continue

            destination = np.array(
                [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                dtype="float32",
            )
            transform = cv2.getPerspectiveTransform(ordered, destination)
            rectified = cv2.warpPerspective(plate_crop, transform, (width, height))
            if rectified.size:
                return rectified
        return plate_crop

    def __init__(self, min_confidence: float = 0.55, model_path: Optional[str] = None):
        self.min_confidence = min_confidence
        self.model = None
        self._track_cache: Dict[int, Tuple[float, PlateResult, List[str]]] = {}
        self._recognition_interval_seconds = float(os.getenv("ANPR_RECOGNITION_INTERVAL", "1.0"))
        self.inference_size = max(320, min(640, int(os.getenv("ANPR_IMGSZ", "320"))))
        # EasyOCR creates another PyTorch neural network beside YOLO and can
        # exceed small Render instance limits. It remains opt-in for larger
        # deployments; production uses the lightweight Tesseract stage.
        self.use_easyocr = os.getenv("ANPR_USE_EASYOCR", "0").strip().lower() in {"1", "true", "yes"}
        self._tesseract_ready = False
        self.model_load_error = None
        try:
            import pytesseract
            self._tesseract_ready = _configure_tesseract(pytesseract)
        except ModuleNotFoundError:
            pass
        if model_path:
            p = Path(model_path)
            if p.is_file():
                try:
                    if p.suffix.lower() == ".onnx":
                        self.model = ONNXPlateDetector(p, self.inference_size)
                    else:
                        from ultralytics import YOLO
                        self.model = YOLO(str(p))
                    logger.info("ANPR YOLO plate detector loaded: %s (%s bytes)", p.resolve(), p.stat().st_size)
                except Exception as exc:
                    self.model = None
                    self.model_load_error = f"{type(exc).__name__}: {exc}"
                    logger.exception(
                        "ANPR YOLO plate detector failed to load: path=%s exists=%s size=%s",
                        p.resolve(), p.is_file(), p.stat().st_size if p.is_file() else None,
                    )
            else:
                self.model_load_error = f"Model file does not exist: {p.resolve()}"
                logger.error("ANPR YOLO plate detector file missing: %s", p.resolve())

    def process_vehicle_crop(
        self,
        vehicle_track_id: int,
        vehicle_image_array: Optional[Any],
        simulated_plate_hint: Optional[str] = None,
        use_temporal_cache: bool = True,
    ) -> PlateResult:
        """
        Executes full ANPR pipeline:
        Detection -> Crop -> Enhancement -> OCR -> Validation -> Confidence Calibration.
        """
        now = time.monotonic()
        cached = self._track_cache.get(vehicle_track_id)
        if use_temporal_cache and vehicle_image_array is not None and cached and now - cached[0] < self._recognition_interval_seconds:
            return cached[1]
        if vehicle_image_array is not None:
            raw_text, det_conf, ocr_conf, plate_bbox, localizer_status, timings = self._cv_ocr_pipeline(vehicle_image_array)
        elif simulated_plate_hint:
            raw_text = simulated_plate_hint
            det_conf = 0.93
            ocr_conf = 0.88
            plate_bbox = None
            localizer_status = "simulated_hint"
            timings = {}
        else:
            raw_text = ""
            det_conf = 0.0
            ocr_conf = 0.0
            plate_bbox = None
            localizer_status = "not_run"
            timings = {}

        validation_started = time.perf_counter()
        # Clean string: uppercase alphanumeric only
        cleaned_text = re.sub(r"[^A-Z0-9]", "", raw_text.upper())
        if cleaned_text.startswith("IND") and len(cleaned_text) >= 8:
            candidate = cleaned_text[3:]
            if self.INDIAN_PLATE_REGEX.match(candidate):
                cleaned_text = candidate

        # Validate syntax against official Indian vehicle format
        is_valid = bool(self.INDIAN_PLATE_REGEX.match(cleaned_text))
        timings["validation_ms"] = round((time.perf_counter() - validation_started) * 1000, 2)
        
        # Overall confidence is geometric mean of plate localization and character recognition
        overall_conf = round((det_conf * ocr_conf) ** 0.5, 3) if det_conf else round(ocr_conf, 3)

        # Guard against hallucination
        is_low_conf = det_conf <= 0 or overall_conf < self.min_confidence or len(cleaned_text) < 6
        final_plate_number = cleaned_text if (not is_low_conf and is_valid) else ("UNKNOWN" if is_low_conf else f"{cleaned_text} (UNVERIFIED)")
        requires_manual = is_low_conf or not is_valid or vehicle_image_array is not None

        explainability = [
            f"Plate region detected with {int(det_conf * 100)}% visual localization confidence",
            f"Character OCR text extraction score: {int(ocr_conf * 100)}%",
            f"Format Syntax Match: {'VALID (Indian Motor Vehicle Standard)' if is_valid else 'NON-STANDARD / IRREGULAR'}"
        ]
        if is_low_conf:
            explainability.append("LOW CONFIDENCE WARNING: OCR quality below safety threshold; plate marked UNKNOWN to prevent erroneous enforcement")
        if requires_manual:
            explainability.append("Flagged for manual operator verification in Command Center")

        result = PlateResult(
            plate_id=f"plt_{uuid.uuid4().hex[:8]}",
            vehicle_track_id=vehicle_track_id,
            plate_number=final_plate_number,
            raw_ocr_text=cleaned_text,
            plate_detection_confidence=round(det_conf, 2),
            ocr_confidence=round(ocr_conf, 2),
            overall_confidence=overall_conf,
            is_format_valid=is_valid,
            is_low_confidence=is_low_conf,
            requires_manual_verification=requires_manual,
            explainability=explainability,
            plate_bbox=plate_bbox,
            localizer_status=localizer_status,
            timings=timings,
        )
        if use_temporal_cache and vehicle_image_array is not None:
            observations = cached[2] if cached else []
            if result.is_format_valid and result.raw_ocr_text:
                observations = (observations + [result.raw_ocr_text])[-5:]
                result.explainability.append(f"Temporal observations retained: {len(observations)}")
                if len(observations) >= 3:
                    result.raw_ocr_text = max(set(observations), key=observations.count)
                    result.is_format_valid = bool(self.INDIAN_PLATE_REGEX.match(result.raw_ocr_text))
            self._track_cache[vehicle_track_id] = (time.monotonic(), result, observations)
        return result

    def _cv_ocr_pipeline(self, frame_crop: Any) -> Tuple[str, float, float]:
        """Runs OpenCV enhancement and OCR if Tesseract / EasyOCR / PaddleOCR is installed."""
        started = time.perf_counter()
        timings: Dict[str, float] = {}
        plate_bbox = None
        localizer_status = "localizer_failure"
        try:
            import cv2
            import numpy as np
            plate_crop = frame_crop
            detection_confidence = 0.0
            if self.model is not None:
                try:
                    detection_started = time.perf_counter()
                    if hasattr(self.model, "detect"):
                        candidates = self.model.detect(frame_crop, confidence=0.20)
                    else:
                        results = self.model(
                            frame_crop,
                            conf=0.20,
                            verbose=False,
                            imgsz=self.inference_size,
                            max_det=10,
                            device="cpu",
                        )
                        candidates = [
                            {"confidence": float(box.conf[0]), "xyxy": box.xyxy[0].tolist()}
                            for result in results for box in result.boxes
                        ]
                    timings["plate_detection_ms"] = round((time.perf_counter() - detection_started) * 1000, 2)
                    if candidates:
                        best = max(candidates, key=lambda item: item["confidence"])
                        x1, y1, x2, y2 = [int(value) for value in best["xyxy"]]
                        height, width = frame_crop.shape[:2]
                        # Expand box slightly (5%) to avoid cutting off plate borders
                        pad_x = int((x2 - x1) * 0.05)
                        pad_y = int((y2 - y1) * 0.05)
                        x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
                        x2, y2 = min(width, x2 + pad_x), min(height, y2 + pad_y)
                        if x2 > x1 and y2 > y1:
                            plate_crop = frame_crop[y1:y2, x1:x2]
                            detection_confidence = float(best["confidence"])
                            plate_bbox = [x1, y1, x2, y2]
                            localizer_status = "localized"
                    else:
                        # Do not initialize a second neural network (EasyOCR) or
                        # OCR an entire vehicle/scene when the trained localizer
                        # found no plate. This keeps negative scans lightweight.
                        timings["ocr_ms"] = 0.0
                        timings["validation_ms"] = 0.0
                        timings["total_ms"] = round((time.perf_counter() - started) * 1000, 2)
                        return "", 0.0, 0.0, None, "no_plate_detected", timings
                except Exception:
                    timings.setdefault("plate_detection_ms", 0.0)
            crop_started = time.perf_counter()
            plate_crop = self._rectify_plate_crop(plate_crop, cv2, np)
            # Tesseract is substantially more reliable after perspective
            # correction and upscaling of small plate characters.
            scale = min(3.0, max(1.0, 480.0 / max(plate_crop.shape[0], 1)))
            if scale > 1.0:
                plate_crop = cv2.resize(
                    plate_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
                )
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            denoised = cv2.bilateralFilter(gray, 7, 35, 35)
            enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(denoised)
            _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            timings["crop_enhancement_ms"] = round((time.perf_counter() - crop_started) * 1000, 2)
            # Local fallback when Tesseract happens to be installed.
            try:
                import pytesseract
                if self._tesseract_ready:
                    ocr_started = time.perf_counter()
                    tess_candidates = []
                    # Do not apply the whitelist to the primary passes. With
                    # Tesseract 5 it can merge all tokens into one zero-score
                    # token, even when the characters themselves are correct.
                    for img, psm in (
                        (plate_crop, "--psm 6"),
                        (enhanced, "--psm 6"),
                        (thresh, "--psm 6"),
                        (enhanced, "--psm 11"),
                    ):
                        data = pytesseract.image_to_data(
                            img,
                            config=f"--oem 3 {psm}",
                            output_type=pytesseract.Output.DICT,
                        )
                        tokens = [(text.strip(), float(conf)) for text, conf in zip(data["text"], data["conf"]) if text.strip() and float(conf) >= 0]
                        if tokens:
                            text = "".join(t for t, _ in tokens)
                            conf = sum(c for _, c in tokens) / (100 * len(tokens))
                            tess_candidates.append((text, conf))
                    if tess_candidates:
                        selected = self._select_ocr_candidate(tess_candidates)
                        text, confidence = selected
                        timings["ocr_ms"] = round((time.perf_counter() - ocr_started) * 1000, 2)
                        timings["validation_ms"] = 0.0
                        timings["total_ms"] = round((time.perf_counter() - started) * 1000, 2)
                        return text, detection_confidence, confidence, plate_bbox, localizer_status, timings
            except Exception:
                pass
            # EasyOCR is accurate but heavy on small production instances. Only
            # use it when explicitly enabled or when a plate detector is present.
            if self.use_easyocr:
                try:
                    import easyocr
                    if not hasattr(self, "_easyocr_reader"):
                        self._easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
                    candidates = []
                    for candidate in (plate_crop, thresh, gray):
                        reads = self._easyocr_reader.readtext(
                            candidate, detail=1,
                            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                            paragraph=False,
                        )
                        if reads:
                            text = "".join(str(item[1]).strip() for item in reads)
                            confidence = sum(float(item[2]) for item in reads) / len(reads)
                            candidates.append((text, confidence))
                    if candidates:
                        text, confidence = max(candidates, key=lambda item: item[1])
                        if detection_confidence <= 0:
                            detection_confidence = confidence
                        timings["ocr_ms"] = round((time.perf_counter() - started) * 1000, 2)
                        timings["validation_ms"] = 0.0
                        timings["total_ms"] = round((time.perf_counter() - started) * 1000, 2)
                        return text, detection_confidence, confidence, plate_bbox, localizer_status, timings
                except ModuleNotFoundError:
                    pass
                except Exception as exc:
                    import logging
                    logging.getLogger("uvicorn.error").warning("EasyOCR ANPR failed: %s", exc)
        except Exception:
            pass
        timings.setdefault("ocr_ms", round((time.perf_counter() - started) * 1000, 2))
        timings["validation_ms"] = 0.0
        timings["total_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return "", detection_confidence, 0.0, plate_bbox, localizer_status, timings
