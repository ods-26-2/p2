from pathlib import Path
from typing import Final

import pytest

from helpers.model_package import write_package
from p2.api import BACKEND_ENV
from p2.cli.build import EXIT_OK, main
from p2.manifest import ENGINE_DIR_ENV

YOLO_FIXTURE: Final = Path(__file__).parent.parent / "fixtures" / "yolo11n-1.0.0"


@pytest.fixture(scope="session")
def compiled_synth_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    pytest.importorskip("tensorrt")
    base = tmp_path_factory.mktemp("synth")
    package = write_package(base / "pkg")
    root = base / "engines"
    assert main([str(package), "--engine-dir", str(root)]) == EXIT_OK
    return root


@pytest.fixture(scope="session")
def compiled_yolo_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    pytest.importorskip("tensorrt")
    if not (YOLO_FIXTURE / "model.onnx").is_file():
        pytest.skip("model.onnx do yolo11n ausente")
    root = tmp_path_factory.mktemp("yolo") / "engines"
    assert main([str(YOLO_FIXTURE), "--engine-dir", str(root)]) == EXIT_OK
    return root


@pytest.fixture
def trt_env(compiled_synth_root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv(ENGINE_DIR_ENV, str(compiled_synth_root))
    monkeypatch.setenv(BACKEND_ENV, "trt")
    return compiled_synth_root
