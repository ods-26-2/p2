import hashlib
from pathlib import Path

import pytest

from p2.backends.variants import memory_mb, read_engine, select_variant
from p2.errors import ErrorCode, P2Error
from p2.manifest import Manifest, VariantInfo


def _manifest(*, tensorrt: str = "10.3.0", with_variants: bool = True) -> Manifest:
    variants: dict[str, VariantInfo] = {
        "FP16": {"file": "fp16.engine", "engine_version": "m-1-fp16", "sha256": "a" * 64},
    }
    return {
        "schema_version": 1,
        "engine_ref": "m-1",
        "source": {"onnx_file": "m.onnx", "sha256": "b" * 64, "metadata_file": "metadata.json"},
        "inputs": {"images": {"shape": [1, 3, 8, 8], "dtype": "float32"}},
        "outputs": {"output0": {"shape": [1, 7, 4], "dtype": "float32"}},
        "max_batch": 1,
        "variants": {"FP16": variants["FP16"]} if with_variants else {},
        "platform": {"tensorrt": tensorrt} if with_variants else {},
        "build": {"tool": "p2-build 0.1.0", "built_at": "x", "tf32": False, "workspace_mb": 1024},
    }


def test_select_existing_variant() -> None:
    variant = select_variant(_manifest(), "FP16", "10.3.0.30")
    assert variant["file"] == "fp16.engine"


def test_cpu_only_dir_is_engine_not_found() -> None:
    with pytest.raises(P2Error, match=r"--cpu-only") as error:
        select_variant(_manifest(with_variants=False), "FP16", "10.3.0")
    assert error.value.code == ErrorCode.ENGINE_NOT_FOUND


def test_missing_variant_is_engine_not_found() -> None:
    with pytest.raises(P2Error, match=r"FP32 .* não compilada; disponíveis: FP16") as error:
        select_variant(_manifest(), "FP32", "10.3.0")
    assert error.value.code == ErrorCode.ENGINE_NOT_FOUND


@pytest.mark.parametrize("installed", ["10.4.0", "10.3.1.5", "8.6.1"])
def test_other_tensorrt_is_incompatible(installed: str) -> None:
    with pytest.raises(P2Error, match="recompile com p2-build --force") as error:
        select_variant(_manifest(), "FP16", installed)
    assert error.value.code == ErrorCode.INCOMPATIBLE_VERSION


def test_read_engine_checks_hash(tmp_path: Path) -> None:
    data = b"engine-bytes"
    (tmp_path / "fp16.engine").write_bytes(data)
    variant: VariantInfo = {
        "file": "fp16.engine",
        "engine_version": "m-1-fp16",
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    assert read_engine(tmp_path, variant) == data


def test_read_engine_corrupt(tmp_path: Path) -> None:
    (tmp_path / "fp16.engine").write_bytes(b"outro")
    variant: VariantInfo = {"file": "fp16.engine", "engine_version": "x", "sha256": "0" * 64}
    with pytest.raises(P2Error, match="engine corrompido") as error:
        read_engine(tmp_path, variant)
    assert error.value.code == ErrorCode.ENGINE_NOT_FOUND


def test_read_engine_missing(tmp_path: Path) -> None:
    variant: VariantInfo = {"file": "fp16.engine", "engine_version": "x", "sha256": "0" * 64}
    with pytest.raises(P2Error, match="engine ausente") as error:
        read_engine(tmp_path, variant)
    assert error.value.code == ErrorCode.ENGINE_NOT_FOUND


def test_memory_mb() -> None:
    mb = 1024 * 1024
    assert memory_mb(engine_bytes=mb, device_memory=2 * mb, buffer_bytes=mb // 2) == 3.5  # noqa: PLR2004
