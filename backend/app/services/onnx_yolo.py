"""Minimal ONNX Runtime wrapper for exported Ultralytics detection models.

This keeps CPU inference independent from PyTorch at request time.  The
trained checkpoint is exported during the Docker build; no weights or scores
are changed here.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class ONNXYOLODetector:
    """Run a fixed-size YOLO detect/segment export with class-aware NMS."""

    def __init__(self, model_path: Path):
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
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        shape = model_input.shape
        self.input_height = int(shape[-2])
        self.input_width = int(shape[-1])
        metadata = self.session.get_modelmeta().custom_metadata_map
        parsed_names = ast.literal_eval(metadata.get("names", "{}"))
        self.names = {int(key): str(value) for key, value in parsed_names.items()}
        if not self.names:
            raise ValueError(f"ONNX model has no class metadata: {model_path}")

    def predict(
        self,
        source: np.ndarray,
        conf: float,
        iou: float = 0.45,
        max_det: int = 200,
        **_: Any,
    ) -> list[dict[str, Any]]:
        """Return API-neutral detections in source-image coordinates."""
        height, width = source.shape[:2]
        scale = min(self.input_width / width, self.input_height / height)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        resized = cv2.resize(
            source,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )
        pad_x = (self.input_width - resized_width) // 2
        pad_y = (self.input_height - resized_height) // 2
        canvas = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
        canvas[pad_y:pad_y + resized_height, pad_x:pad_x + resized_width] = resized
        tensor = canvas.transpose(2, 0, 1).astype(np.float32) / 255.0
        output = self.session.run(None, {self.input_name: tensor[None]})[0]
        prediction = np.squeeze(output)
        if prediction.ndim != 2:
            return []
        class_count = len(self.names)
        if prediction.shape[0] == 4 + class_count or prediction.shape[0] < prediction.shape[1]:
            prediction = prediction.T

        if prediction.shape[1] < 4 + class_count:
            return []
        class_scores = prediction[:, 4:4 + class_count]
        class_ids = class_scores.argmax(axis=1)
        scores = class_scores[np.arange(class_scores.shape[0]), class_ids]
        selected = np.flatnonzero(scores >= conf)
        if not selected.size:
            return []

        candidates: list[dict[str, Any]] = []
        for class_id in sorted(set(int(class_ids[index]) for index in selected)):
            class_indexes = [
                int(index) for index in selected if int(class_ids[index]) == class_id
            ]
            boxes_xywh = []
            boxes_xyxy = []
            box_scores = []
            for index in class_indexes:
                center_x, center_y, box_width, box_height = prediction[index, :4]
                x1 = float(center_x - box_width / 2)
                y1 = float(center_y - box_height / 2)
                boxes_xywh.append([int(x1), int(y1), int(box_width), int(box_height)])
                boxes_xyxy.append(
                    [x1, y1, float(center_x + box_width / 2), float(center_y + box_height / 2)]
                )
                box_scores.append(float(scores[index]))

            kept = cv2.dnn.NMSBoxes(boxes_xywh, box_scores, conf, iou)
            for raw_index in np.asarray(kept).reshape(-1):
                index = int(raw_index)
                x1, y1, x2, y2 = boxes_xyxy[index]
                candidates.append({
                    "class_id": class_id,
                    "class_name": self.names[class_id],
                    "confidence": box_scores[index],
                    "xyxy": [
                        max(0.0, min(float(width), (x1 - pad_x) / scale)),
                        max(0.0, min(float(height), (y1 - pad_y) / scale)),
                        max(0.0, min(float(width), (x2 - pad_x) / scale)),
                        max(0.0, min(float(height), (y2 - pad_y) / scale)),
                    ],
                })

        return sorted(candidates, key=lambda item: item["confidence"], reverse=True)[:max_det]
