from pathlib import Path

import pytest

from helpers.dynamic_package import write_dynamic_package
from helpers.model_package import write_package
from helpers.yolo11n import YOLO11N_MISSING, yolo11n_available, yolo11n_package
from p2.api import BACKEND_ENV
from p2.cli.build import EXIT_OK, main
from p2.manifest import ENGINE_DIR_ENV


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
    if not yolo11n_available():
        pytest.skip(YOLO11N_MISSING)
    base = tmp_path_factory.mktemp("yolo")
    package = yolo11n_package(base / "pkg")
    root = base / "engines"
    assert main([str(package), "--engine-dir", str(root)]) == EXIT_OK
    return root


@pytest.fixture(scope="session")
def compiled_dynamic_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    pytest.importorskip("tensorrt")
    base = tmp_path_factory.mktemp("dynamic")
    package = write_dynamic_package(base / "pkg")
    root = base / "engines"
    assert main([str(package), "--engine-dir", str(root)]) == EXIT_OK
    return root


@pytest.fixture
def trt_env(compiled_synth_root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv(ENGINE_DIR_ENV, str(compiled_synth_root))
    monkeypatch.setenv(BACKEND_ENV, "trt")
    return compiled_synth_root
