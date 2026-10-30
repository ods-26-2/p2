import json
import re
from importlib.metadata import version
from pathlib import Path
from typing import Final

import cv2
import numpy as np
import pytest

import p2
from helpers.yolo import detect, letterbox
from p2.cli.build import EXIT_OK, main
from p2.errors import ErrorCode
from p2.manifest import ENGINE_DIR_ENV, MANIFEST_FILE

FIXTURE_DIR: Final = Path(__file__).parent / "fixtures" / "yolo11n-1.0.0"
ENGINE_REF: Final = "yolo11n-1.0.0"
BUS_IMAGE: Final = Path(__file__).parent.parent / "assets" / "bus.jpg"
PERSON: Final = 0
MIN_PERSON_CONF: Final = 0.5
OUTPUT_SHAPE: Final = (1, 84, 8400)
ISO_UTC: Final = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00$")

pytestmark = pytest.mark.skipif(
    not (FIXTURE_DIR / "model.onnx").is_file(),
    reason="model.onnx do yolo11n ausente em tests/fixtures/yolo11n-1.0.0",
)


def test_cpu_only_build_matches_expected_manifest(tmp_path: Path) -> None:
    exit_code = main([str(FIXTURE_DIR), "--cpu-only", "--engine-dir", str(tmp_path)])
    assert exit_code == EXIT_OK
    loaded: object = json.loads((tmp_path / ENGINE_REF / MANIFEST_FILE).read_text("utf-8"))
    assert isinstance(loaded, dict)
    build = loaded.pop("build")
    assert loaded == {
        "schema_version": 1,
        "engine_ref": ENGINE_REF,
        "source": {
            "onnx_file": "model.onnx",
            "sha256": "f3f88a57b89f447e639b86279cdb0c636719f8bf7fefab8f3e27c44536e48de1",
            "metadata_file": "metadata.json",
        },
        "inputs": {"images": {"shape": [1, 3, 640, 640], "dtype": "float32"}},
        "outputs": {"output0": {"shape": [1, 84, 8400], "dtype": "float32"}},
        "max_batch": 1,
        "variants": {},
        "platform": {},
    }
    assert isinstance(build, dict)
    assert build["tool"] == f"p2-build {version('p2')}"
    assert ISO_UTC.match(build["built_at"])
    assert build["tf32"] is False
    assert build["workspace_mb"] == 1024  # noqa: PLR2004


@pytest.fixture
def yolo_engine_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    assert main([str(FIXTURE_DIR), "--cpu-only", "--engine-dir", str(tmp_path)]) == EXIT_OK
    monkeypatch.setenv(ENGINE_DIR_ENV, str(tmp_path))
    monkeypatch.delenv("P2_BACKEND", raising=False)
    return tmp_path


@pytest.mark.usefixtures("yolo_engine_root")
def test_load_info_on_cpu() -> None:
    with p2.load(ENGINE_REF) as handle:
        assert p2.info(handle) == {
            "engine_version": "yolo11n-1.0.0-fp32",
            "input_shape": {"images": [1, 3, 640, 640]},
            "output_shape": {"output0": list(OUTPUT_SHAPE)},
            "precision": "FP32",
            "max_batch": 1,
        }


@pytest.mark.usefixtures("yolo_engine_root")
def test_infer_zeros_on_cpu() -> None:
    with p2.load(ENGINE_REF) as handle:
        outputs, stats = p2.infer(handle, np.zeros((1, 3, 640, 640), dtype=np.float32))
    assert outputs["output0"].shape == OUTPUT_SHAPE
    assert outputs["output0"].dtype == np.float32
    assert set(stats) == {"latency_ms", "gpu_memory_mb"}


@pytest.mark.usefixtures("yolo_engine_root")
def test_fp16_on_cpu_is_engine_not_found() -> None:
    with pytest.raises(p2.P2Error) as error:
        p2.load(ENGINE_REF, {"precision": "FP16"})
    assert error.value.code == ErrorCode.ENGINE_NOT_FOUND


@pytest.mark.usefixtures("yolo_engine_root")
def test_detects_person_in_bus_image_on_cpu() -> None:
    image = cv2.imread(str(BUS_IMAGE))
    assert image is not None
    boxed = letterbox(image.astype(np.uint8))
    with p2.load(ENGINE_REF) as handle:
        outputs, _ = p2.infer(handle, boxed.tensor)
    detections = detect(outputs["output0"])
    people = detections.scores[detections.classes == PERSON]
    assert people.size > 0
    assert float(people.max()) >= MIN_PERSON_CONF
