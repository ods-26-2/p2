import json
import shutil
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import pytest

import p2
from helpers.model_package import (
    ANCHORS,
    CLASSES,
    ENGINE_REF,
    INPUT_NAME,
    INPUT_SHAPE,
    OUTPUT_NAME,
    build_engine,
)
from p2.errors import ErrorCode
from p2.manifest import ENGINE_DIR_ENV, MANIFEST_FILE

pytest.importorskip("tensorrt")

pytest.importorskip("cuda.cudart")

from cuda import cudart

pytestmark = pytest.mark.jetson

OUTPUT_SHAPE: Final = (1, 4 + len(CLASSES), ANCHORS)
FP16_TOLERANCE: Final = 0.05
THREADS: Final = 4


def _input(value: float = 0.0) -> npt.NDArray[np.float32]:
    return np.full(INPUT_SHAPE, value, dtype=np.float32)


def _expected(value: float) -> npt.NDArray[np.float32]:
    base = np.arange(np.prod(OUTPUT_SHAPE), dtype=np.float32).reshape(OUTPUT_SHAPE)
    return base + np.float32(value)


def _copy_with(
    root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: Callable[[dict[str, object]], None],
) -> None:
    copy_root = tmp_path / "engines"
    shutil.copytree(root, copy_root)
    manifest_path = copy_root / ENGINE_REF / MANIFEST_FILE
    manifest: dict[str, object] = json.loads(manifest_path.read_text(encoding="utf-8"))
    change(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setenv(ENGINE_DIR_ENV, str(copy_root))


@pytest.mark.usefixtures("trt_env")
def test_default_is_fp16_on_gpu() -> None:
    with p2.load(ENGINE_REF) as handle:
        info = p2.info(handle)
        outputs, stats = p2.infer(handle, _input(2.0))
    assert info["precision"] == "FP16"
    assert info["engine_version"] == f"{ENGINE_REF}-fp16"
    assert outputs[OUTPUT_NAME].shape == OUTPUT_SHAPE
    np.testing.assert_allclose(outputs[OUTPUT_NAME], _expected(2.0), atol=FP16_TOLERANCE)
    assert stats["gpu_memory_mb"] > 0.0
    assert stats["latency_ms"] > 0.0


@pytest.mark.usefixtures("trt_env")
def test_fp32_on_gpu() -> None:
    with p2.load(ENGINE_REF, {"precision": "FP32"}) as handle:
        assert p2.info(handle)["engine_version"] == f"{ENGINE_REF}-fp32"
        outputs, stats = p2.infer(handle, {INPUT_NAME: _input(3.0)})
    np.testing.assert_allclose(outputs[OUTPUT_NAME], _expected(3.0), rtol=1e-6)
    assert stats["gpu_memory_mb"] > 0.0


@pytest.mark.usefixtures("trt_env")
def test_gpu_memory_is_constant_per_handle() -> None:
    with p2.load(ENGINE_REF) as handle:
        first = p2.infer(handle, _input())[1]["gpu_memory_mb"]
        second = p2.infer(handle, _input(1.0))[1]["gpu_memory_mb"]
    assert first == second


@pytest.mark.usefixtures("trt_env")
def test_outputs_are_copies_on_gpu() -> None:
    with p2.load(ENGINE_REF, {"precision": "FP32"}) as handle:
        first, _ = p2.infer(handle, _input())
        first[OUTPUT_NAME][...] = -1.0
        second, _ = p2.infer(handle, _input())
    np.testing.assert_allclose(second[OUTPUT_NAME], _expected(0.0), rtol=1e-6)


@pytest.mark.usefixtures("trt_env")
def test_concurrent_calls_on_gpu() -> None:
    errors: list[BaseException] = []
    with p2.load(ENGINE_REF, {"precision": "FP32"}) as handle:

        def worker(value: float) -> None:
            try:
                for _ in range(10):
                    outputs, _ = p2.infer(handle, _input(value))
                    np.testing.assert_allclose(outputs[OUTPUT_NAME], _expected(value), rtol=1e-6)
            except (AssertionError, p2.P2Error) as err:
                errors.append(err)

        threads = [threading.Thread(target=worker, args=(float(i),)) for i in range(THREADS)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    assert errors == []


@pytest.mark.usefixtures("trt_env")
def test_release_then_use_is_value_error() -> None:
    handle = p2.load(ENGINE_REF)
    p2.release(handle)
    p2.release(handle)
    with pytest.raises(ValueError, match="já foi liberado"):
        p2.infer(handle, _input())


def test_missing_variant_is_engine_not_found(
    trt_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def drop_fp16(manifest: dict[str, object]) -> None:
        variants = manifest["variants"]
        assert isinstance(variants, dict)
        del variants["FP16"]

    _copy_with(trt_env, tmp_path, monkeypatch, drop_fp16)
    with pytest.raises(p2.P2Error, match="FP16") as error:
        p2.load(ENGINE_REF)
    assert error.value.code == ErrorCode.ENGINE_NOT_FOUND


def test_other_tensorrt_is_incompatible_version(
    trt_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def other_version(manifest: dict[str, object]) -> None:
        platform = manifest["platform"]
        assert isinstance(platform, dict)
        platform["tensorrt"] = "9.9.9"

    _copy_with(trt_env, tmp_path, monkeypatch, other_version)
    with pytest.raises(p2.P2Error, match=r"9\.9\.9") as error:
        p2.load(ENGINE_REF)
    assert error.value.code == ErrorCode.INCOMPATIBLE_VERSION


def test_corrupt_engine_is_engine_not_found(
    trt_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _copy_with(trt_env, tmp_path, monkeypatch, lambda manifest: None)
    (tmp_path / "engines" / ENGINE_REF / "fp16.engine").write_bytes(b"corrompido")
    with pytest.raises(p2.P2Error, match="engine corrompido") as error:
        p2.load(ENGINE_REF)
    assert error.value.code == ErrorCode.ENGINE_NOT_FOUND


@pytest.mark.usefixtures("trt_env")
def test_malloc_failure_is_out_of_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing_malloc(size: int) -> tuple[cudart.cudaError_t, int]:
        return cudart.cudaError_t.cudaErrorMemoryAllocation, 0

    monkeypatch.setattr(cudart, "cudaMalloc", failing_malloc)
    with pytest.raises(p2.P2Error, match="cudaMalloc") as error:
        p2.load(ENGINE_REF)
    assert error.value.code == ErrorCode.OUT_OF_MEMORY


@pytest.mark.usefixtures("trt_env")
def test_invalid_input_on_gpu() -> None:
    with p2.load(ENGINE_REF) as handle, pytest.raises(p2.P2Error) as error:
        p2.infer(handle, np.zeros((1, 3, 8, 7), dtype=np.float32))
    assert error.value.code == ErrorCode.INVALID_INPUT_SHAPE


@pytest.mark.usefixtures("trt_env")
def test_copy_failure_is_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    with p2.load(ENGINE_REF) as handle:

        def failing_copy(*args: object) -> tuple[cudart.cudaError_t]:
            return (cudart.cudaError_t.cudaErrorMemoryAllocation,)

        monkeypatch.setattr(cudart, "cudaMemcpyAsync", failing_copy)
        with pytest.raises(p2.P2Error, match="cópia da entrada") as error:
            p2.infer(handle, _input())
    assert error.value.code == ErrorCode.RUNTIME_ERROR


def test_cpu_only_dir_with_trt_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    build_engine(tmp_path / "pkg", tmp_path / "engines")
    monkeypatch.setenv(ENGINE_DIR_ENV, str(tmp_path / "engines"))
    monkeypatch.setenv("P2_BACKEND", "trt")
    with pytest.raises(p2.P2Error, match="--cpu-only") as error:
        p2.load(ENGINE_REF)
    assert error.value.code == ErrorCode.ENGINE_NOT_FOUND


@pytest.mark.usefixtures("trt_env")
def test_automatic_backend_prefers_tensorrt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("P2_BACKEND")
    with p2.load(ENGINE_REF) as handle:
        assert p2.info(handle)["precision"] == "FP16"
        assert p2.infer(handle, _input())[1]["gpu_memory_mb"] > 0.0
