import numpy as np

from app.services.onnx_yolo import ONNXYOLODetector


class _Input:
    name = "images"
    shape = [1, 3, 64, 64]


class _Metadata:
    custom_metadata_map = {"names": "{0: 'pothole', 1: 'waterlogging'}"}


class _Session:
    def get_inputs(self):
        return [_Input()]

    def get_modelmeta(self):
        return _Metadata()

    def run(self, _outputs, feed):
        assert feed["images"].shape == (1, 3, 64, 64)
        # [xywh + two class scores, anchors]
        return [np.array([[
            [32.0, 32.0], [32.0, 32.0], [20.0, 10.0], [20.0, 10.0],
            [0.90, 0.10], [0.05, 0.85],
        ]], dtype=np.float32)]


def test_onnx_detector_preserves_classes_and_multiple_boxes():
    detector = ONNXYOLODetector.__new__(ONNXYOLODetector)
    detector.session = _Session()
    detector.input_name = "images"
    detector.input_height = 64
    detector.input_width = 64
    detector.names = {0: "pothole", 1: "waterlogging"}

    detections = detector.predict(np.zeros((64, 64, 3), dtype=np.uint8), conf=0.20)

    assert [item["class_name"] for item in detections] == ["pothole", "waterlogging"]
    assert [round(item["confidence"], 2) for item in detections] == [0.90, 0.85]
