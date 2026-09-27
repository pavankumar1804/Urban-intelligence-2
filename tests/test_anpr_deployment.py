"""Focused ANPR deployment/integration tests."""
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from backend.app.services import anpr_service


def _image_bytes():
    buf = BytesIO()
    Image.new("RGB", (32, 16), "white").save(buf, format="JPEG")
    return buf.getvalue()


def test_backend_anpr_weight_is_deployed_and_matches_frontend_copy():
    repo = Path(__file__).resolve().parents[1]
    backend_weight = repo / "backend" / "app" / "weights" / "anpr_plate.pt"
    frontend_weight = repo / "frontend" / "ml" / "weights" / "anpr_plate.pt"
    assert backend_weight.is_file()
    assert frontend_weight.is_file()
    assert backend_weight.stat().st_size == frontend_weight.stat().st_size
    assert backend_weight.read_bytes() == frontend_weight.read_bytes()


def test_backend_weight_has_priority():
    candidates = anpr_service._weight_candidates("anpr_plate.pt")
    assert candidates[0].name == "anpr_plate.pt"
    assert candidates[0].parent.name == "weights"
    assert candidates[0].parent.parent.name == "app"
    assert anpr_service.WEIGHTS == candidates[0]


def test_processor_receives_discovered_trained_weight(monkeypatch):
    received = {}

    class LoadedProcessor:
        model = object()
        model_load_error = None

        def __init__(self, model_path=None):
            received["model_path"] = model_path

    monkeypatch.setattr(anpr_service, "ANPRProcessor", LoadedProcessor)
    monkeypatch.setattr(anpr_service, "_processor", None)
    monkeypatch.setattr(anpr_service, "_processor_error", None)
    monkeypatch.setattr(anpr_service, "_processor_load_ms", None)

    processor = anpr_service.get_processor()
    health = anpr_service.anpr_model_health()

    assert processor.model is not None
    assert Path(received["model_path"]).resolve() == anpr_service.WEIGHTS.resolve()
    assert health["anpr_model_ready"] is True
    assert health["anpr_weight"] == "anpr_plate.pt"
    assert health["model_cached"] is True
    assert health["anpr_error"] is None


def test_model_load_failure_surfaces_processor_error(monkeypatch):
    class FailedProcessor:
        model = None
        model_load_error = "RuntimeError: incompatible Torch build"

        def __init__(self, model_path=None):
            pass

    monkeypatch.setattr(anpr_service, "ANPRProcessor", FailedProcessor)
    monkeypatch.setattr(anpr_service, "_processor", None)
    monkeypatch.setattr(anpr_service, "_processor_error", None)
    monkeypatch.setattr(anpr_service, "_processor_load_ms", None)

    processor = anpr_service.get_processor()
    health = anpr_service.anpr_model_health()

    assert processor.model is None
    assert health["anpr_model_ready"] is False
    assert health["anpr_error"] == "RuntimeError: incompatible Torch build"


def test_loaded_detector_stays_trained_when_ocr_fails(monkeypatch):
    result = SimpleNamespace(
        plate_number="UNKNOWN", raw_ocr_text="", plate_detection_confidence=0.91,
        ocr_confidence=0.0, overall_confidence=0.0, is_format_valid=False,
        requires_manual_verification=True, timestamp="now", plate_bbox=[1, 1, 10, 8],
        localizer_status="localized", timings={},
    )
    processor = SimpleNamespace(model=object(), process_vehicle_crop=lambda *a, **k: result)
    monkeypatch.setattr(anpr_service, "_processor", processor)
    monkeypatch.setattr(anpr_service, "WEIGHTS", Path("anpr_plate.pt"))
    response = anpr_service.recognize_plate(_image_bytes())
    assert response["mode"] == "trained_plate_detector"
    assert response["warning"] is None
    assert response["anpr_status"] == "plate_detected_ocr_failed"
    assert response["plate_number"] == "UNKNOWN"


def test_model_unavailable_is_honest(monkeypatch):
    result = SimpleNamespace(
        plate_number="UNKNOWN", raw_ocr_text="", plate_detection_confidence=0.0,
        ocr_confidence=0.0, overall_confidence=0.0, is_format_valid=False,
        requires_manual_verification=True, timestamp="now", plate_bbox=None,
        localizer_status="localizer_failure", timings={},
    )
    processor = SimpleNamespace(model=None, process_vehicle_crop=lambda *a, **k: result)
    monkeypatch.setattr(anpr_service, "_processor", processor)
    monkeypatch.setattr(anpr_service, "WEIGHTS", None)
    response = anpr_service.recognize_plate(_image_bytes())
    assert response["mode"] == "ocr_prototype_no_plate_detector"
    assert response["anpr_status"] == "model_unavailable"
    assert response["warning"]
