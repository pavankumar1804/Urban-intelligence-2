"""ML Pipeline & Model Weight Integration Tests.

Validates:
1. Integrity and existence of trained/deployed weights (road defect, ANPR plate localizer, traffic).
2. ONNX export availability for edge deployment.
3. Model loadability via Ultralytics YOLO.
4. Model manifest status matching filesystem state.
5. Dataset scaffolding and validation utility functionality.
"""
import json
import subprocess
import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = REPO_ROOT / "frontend" / "ml" / "weights"
MANIFEST_PATH = REPO_ROOT / "frontend" / "ml" / "model_manifest.json"


def test_road_defect_weights_exist():
    """Verify 50-epoch road defect weights and exported ONNX exist and have valid size."""
    pt_path = WEIGHTS_DIR / "road_defect_best.pt"
    onnx_path = WEIGHTS_DIR / "road_defect_best.onnx"

    assert pt_path.is_file(), f"Missing {pt_path}"
    assert onnx_path.is_file(), f"Missing {onnx_path}"
    assert pt_path.stat().st_size > 10_000_000, "Road defect PT weight file is too small"
    assert onnx_path.stat().st_size > 10_000_000, "Road defect ONNX weight file is too small"


def test_anpr_plate_localizer_weights_exist():
    """Verify license plate localizer weights and exported ONNX exist and have valid size."""
    pt_path = WEIGHTS_DIR / "anpr_plate.pt"
    onnx_path = WEIGHTS_DIR / "anpr_plate.onnx"

    assert pt_path.is_file(), f"Missing {pt_path}"
    assert onnx_path.is_file(), f"Missing {onnx_path}"
    assert pt_path.stat().st_size > 1_000_000, "ANPR plate PT weight file is too small"
    assert onnx_path.stat().st_size > 1_000_000, "ANPR plate ONNX weight file is too small"


def test_traffic_coco_weights_exist():
    """Verify traffic COCO weights exist."""
    pt_path = WEIGHTS_DIR / "traffic_coco.pt"
    assert pt_path.is_file(), f"Missing {pt_path}"
    assert pt_path.stat().st_size > 1_000_000, "Traffic weight file is too small"


def test_models_load_in_ultralytics():
    """Verify all 3 model weights load in Ultralytics YOLO without errors."""
    try:
        from ultralytics import YOLO
        python_cmd = None
    except ImportError:
        # If running in backend.venv without ultralytics, verify with system python
        res = subprocess.run(
            [
                "python", "-c",
                "from ultralytics import YOLO;"
                "m1 = YOLO('frontend/ml/weights/road_defect_best.pt'); assert len(m1.names) == 8;"
                "m2 = YOLO('frontend/ml/weights/anpr_plate.pt'); assert 0 in m2.names;"
                "m3 = YOLO('frontend/ml/weights/traffic_coco.pt'); assert 'car' in m3.names.values();"
                "print('OK')"
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0, f"Ultralytics load check failed: {res.stderr}"
        assert "OK" in res.stdout
        return

    # Direct import path when ultralytics is in current venv
    m_road = YOLO(str(WEIGHTS_DIR / "road_defect_best.pt"))
    assert len(m_road.names) == 8

    m_plate = YOLO(str(WEIGHTS_DIR / "anpr_plate.pt"))
    assert 0 in m_plate.names
    assert "plate" in m_plate.names[0].lower()

    m_traffic = YOLO(str(WEIGHTS_DIR / "traffic_coco.pt"))
    assert "car" in m_traffic.names.values()


def test_anpr_processor_with_localization_model():
    """Verify ANPRProcessor loads plate localizer model and runs on simulated crop."""
    try:
        from ultralytics import YOLO  # noqa: F401
    except ImportError:
        pytest.skip("ultralytics not installed in backend venv; tested in system/edge python")

    import numpy as np
    from edge.processors.anpr import ANPRProcessor

    pt_path = WEIGHTS_DIR / "anpr_plate.pt"
    anpr = ANPRProcessor(model_path=str(pt_path))
    assert anpr.model is not None, "ANPRProcessor failed to initialize YOLO model"

    dummy_crop = np.zeros((200, 300, 3), dtype=np.uint8)
    res = anpr.process_vehicle_crop(
        vehicle_track_id=42,
        vehicle_image_array=dummy_crop,
    )
    assert res.plate_number == "UNKNOWN" or res.is_low_confidence
    assert res.vehicle_track_id == 42
    assert res.requires_manual_verification is True


def test_model_manifest_consistency():
    """Verify model_manifest.json accurately reflects weights and status."""
    assert MANIFEST_PATH.is_file(), f"Missing {MANIFEST_PATH}"
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert manifest["strategy"] == "hybrid"
    assert manifest["road_defect"]["status"] == "trained"
    assert (REPO_ROOT / manifest["road_defect"]["path"]).is_file()

    assert manifest["traffic_and_people"]["status"] == "ready_pretrained"
    assert (REPO_ROOT / manifest["traffic_and_people"]["path"]).is_file()

    assert manifest["number_plate_localizer"]["status"] in ("ready_pretrained", "trained")
    assert (REPO_ROOT / manifest["number_plate_localizer"]["path"]).is_file()


def test_dataset_directories_scaffolded():
    """Verify all 4 target dataset folders have train/val/test splits."""
    datasets_root = REPO_ROOT / "frontend" / "ml" / "datasets"
    targets = ["license_plates", "traffic", "infrastructure", "road_defects"]

    for target in targets:
        for folder in ["images", "labels"]:
            for split in ["train", "val", "test"]:
                dir_path = datasets_root / target / folder / split
                assert dir_path.is_dir(), f"Missing directory: {dir_path}"


