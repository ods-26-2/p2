import hashlib
import importlib.metadata
from pathlib import Path
from typing import Final

import pytest

from helpers.model_package import ENGINE_REF, write_package
from p2.build.platform import short_tensorrt_version
from p2.cli.build import EXIT_BUILD_ERROR, EXIT_OK, main
from p2.manifest import read_manifest

pytest.importorskip("tensorrt")

pytestmark = pytest.mark.jetson

YOLO_FIXTURE: Final = Path(__file__).parent.parent / "fixtures" / "yolo11n-1.0.0"
YOLO_REF: Final = "yolo11n-1.0.0"


def _tensorrt_version() -> str:
    import tensorrt  # noqa: PLC0415

    return short_tensorrt_version(tensorrt.__version__)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_compiles_synthetic_package_in_both_precisions(tmp_path: Path) -> None:
    package = write_package(tmp_path / "pkg")
    engines = tmp_path / "engines"
    assert main([str(package), "--engine-dir", str(engines)]) == EXIT_OK
    engine_dir = engines / ENGINE_REF
    manifest = read_manifest(engine_dir)
    assert set(manifest["variants"]) == {"FP16", "FP32"}
    for precision, variant in manifest["variants"].items():
        engine_path = engine_dir / variant["file"]
        assert engine_path.is_file()
        assert variant["sha256"] == _sha256(engine_path)
        assert variant["engine_version"] == f"{ENGINE_REF}-{precision.lower()}"
    assert manifest["platform"]["tensorrt"] == _tensorrt_version()
    assert manifest["build"]["tf32"] is False


def test_single_precision_option(tmp_path: Path) -> None:
    package = write_package(tmp_path / "pkg")
    engines = tmp_path / "engines"
    assert main([str(package), "--precision", "FP32", "--engine-dir", str(engines)]) == EXIT_OK
    engine_dir = engines / ENGINE_REF
    assert set(read_manifest(engine_dir)["variants"]) == {"FP32"}
    assert not (engine_dir / "fp16.engine").exists()


def test_parser_error_is_build_error(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    package = write_package(tmp_path / "pkg")
    onnx_path = package / "model.onnx"
    model = onnx_path.read_bytes().replace(b"ReduceMean", b"ReduceMeen")
    onnx_path.write_bytes(model)
    metadata = (package / "metadata.json").read_text(encoding="utf-8")
    old_hash = metadata.split('"hash_sha256": "')[1].split('"')[0]
    (package / "metadata.json").write_text(
        metadata.replace(old_hash, _sha256(onnx_path)), encoding="utf-8"
    )
    engines = tmp_path / "engines"
    assert main([str(package), "--engine-dir", str(engines)]) == EXIT_BUILD_ERROR
    assert "parser ONNX do TensorRT falhou" in caplog.text
    assert list(engines.iterdir()) == []


@pytest.mark.skipif(
    not (YOLO_FIXTURE / "model.onnx").is_file(), reason="model.onnx do yolo11n ausente"
)
def test_yolo11n_case_6(tmp_path: Path) -> None:
    engines = tmp_path / "engines"
    assert main([str(YOLO_FIXTURE), "--engine-dir", str(engines)]) == EXIT_OK
    engine_dir = engines / YOLO_REF
    manifest = read_manifest(engine_dir)
    assert (engine_dir / "fp16.engine").is_file()
    assert (engine_dir / "fp32.engine").is_file()
    assert manifest["platform"]["tensorrt"] == "10.3.0"
    assert manifest["inputs"] == {"images": {"shape": [1, 3, 640, 640], "dtype": "float32"}}
    assert manifest["outputs"] == {"output0": {"shape": [1, 84, 8400], "dtype": "float32"}}
    assert manifest["build"]["tool"] == f"p2-build {importlib.metadata.version('p2')}"
    assert {variant["engine_version"] for variant in manifest["variants"].values()} == {
        "yolo11n-1.0.0-fp16",
        "yolo11n-1.0.0-fp32",
    }
