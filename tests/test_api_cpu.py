import subprocess
import sys
import threading
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import pytest

import p2
from helpers.model_package import ANCHORS, CLASSES, ENGINE_REF, INPUT_NAME, INPUT_SHAPE, OUTPUT_NAME

OUTPUT_SHAPE: Final = (1, 4 + len(CLASSES), ANCHORS)
THREADS: Final = 4
CALLS_PER_THREAD: Final = 10


def _input(value: float = 0.0) -> npt.NDArray[np.float32]:
    return np.full(INPUT_SHAPE, value, dtype=np.float32)


def _expected(value: float) -> npt.NDArray[np.float32]:
    base = np.arange(np.prod(OUTPUT_SHAPE), dtype=np.float32).reshape(OUTPUT_SHAPE)
    return base + np.float32(value)


@pytest.mark.usefixtures("synth_engine")
def test_full_cycle() -> None:
    handle = p2.load(ENGINE_REF)
    outputs, stats = p2.infer(handle, _input(2.0))
    assert set(outputs) == {OUTPUT_NAME}
    assert outputs[OUTPUT_NAME].shape == OUTPUT_SHAPE
    assert outputs[OUTPUT_NAME].dtype == np.float32
    np.testing.assert_allclose(outputs[OUTPUT_NAME], _expected(2.0))
    assert set(stats) == {"latency_ms", "gpu_memory_mb"}
    assert stats["latency_ms"] > 0.0
    assert stats["gpu_memory_mb"] == 0.0
    p2.release(handle)


@pytest.mark.usefixtures("synth_engine")
def test_info_on_cpu_defaults_to_fp32() -> None:
    with p2.load(ENGINE_REF) as handle:
        assert p2.info(handle) == {
            "engine_version": f"{ENGINE_REF}-fp32",
            "input_shape": {INPUT_NAME: list(INPUT_SHAPE)},
            "output_shape": {OUTPUT_NAME: list(OUTPUT_SHAPE)},
            "precision": "FP32",
            "max_batch": 1,
        }


@pytest.mark.usefixtures("synth_engine")
def test_explicit_fp32() -> None:
    with p2.load(ENGINE_REF, {"precision": "FP32"}) as handle:
        assert p2.info(handle)["precision"] == "FP32"


@pytest.mark.usefixtures("synth_engine")
def test_info_returns_independent_copy() -> None:
    with p2.load(ENGINE_REF) as handle:
        first = p2.info(handle)
        first["input_shape"][INPUT_NAME].append(99)
        assert p2.info(handle)["input_shape"] == {INPUT_NAME: list(INPUT_SHAPE)}


@pytest.mark.usefixtures("synth_engine")
def test_dict_input_matches_array_input() -> None:
    with p2.load(ENGINE_REF) as handle:
        from_array, _ = p2.infer(handle, _input(1.0))
        from_dict, _ = p2.infer(handle, {INPUT_NAME: _input(1.0)})
        np.testing.assert_array_equal(from_array[OUTPUT_NAME], from_dict[OUTPUT_NAME])


@pytest.mark.usefixtures("synth_engine")
def test_non_contiguous_input_is_accepted() -> None:
    wide = np.full((1, 3, 8, 16), 3.0, dtype=np.float32)
    strided = wide[:, :, :, ::2]
    assert not strided.flags["C_CONTIGUOUS"]
    with p2.load(ENGINE_REF) as handle:
        outputs, _ = p2.infer(handle, strided)
        np.testing.assert_allclose(outputs[OUTPUT_NAME], _expected(3.0))


@pytest.mark.usefixtures("synth_engine")
def test_outputs_are_copies() -> None:
    with p2.load(ENGINE_REF) as handle:
        first, _ = p2.infer(handle, _input())
        first[OUTPUT_NAME][...] = -1.0
        second, _ = p2.infer(handle, _input())
        np.testing.assert_allclose(second[OUTPUT_NAME], _expected(0.0))


@pytest.mark.usefixtures("synth_engine")
def test_with_releases_handle() -> None:
    with p2.load(ENGINE_REF) as handle:
        assert not handle.released
    assert handle.released
    with pytest.raises(ValueError, match="já foi liberado"):
        p2.infer(handle, _input())


@pytest.mark.usefixtures("synth_engine")
def test_release_is_idempotent_and_blocks_use() -> None:
    handle = p2.load(ENGINE_REF)
    p2.release(handle)
    p2.release(handle)
    with pytest.raises(ValueError, match="já foi liberado"):
        p2.info(handle)
    with pytest.raises(ValueError, match="já foi liberado"):
        p2.infer(handle, _input())


@pytest.mark.usefixtures("synth_engine")
def test_concurrent_calls_on_same_handle() -> None:
    errors: list[BaseException] = []
    with p2.load(ENGINE_REF) as handle:

        def worker(value: float) -> None:
            try:
                for _ in range(CALLS_PER_THREAD):
                    outputs, _ = p2.infer(handle, _input(value))
                    np.testing.assert_allclose(outputs[OUTPUT_NAME], _expected(value))
            except (AssertionError, p2.P2Error) as err:
                errors.append(err)

        threads = [threading.Thread(target=worker, args=(float(i),)) for i in range(THREADS)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    assert errors == []


@pytest.mark.parametrize(
    "options",
    [{"batch": 2}, {"precision": "INT8"}, {"precision": "fp16"}, {"precision": None}],
)
@pytest.mark.usefixtures("synth_engine")
def test_invalid_options_raise_value_error(options: dict[str, object]) -> None:
    with pytest.raises(ValueError, match=r"opção desconhecida|precisão inválida"):
        p2.load(ENGINE_REF, options)


@pytest.mark.parametrize("engine_ref", ["", "  ", "../x", "a/b", "."])
def test_invalid_engine_ref_raises_value_error(engine_ref: str, engine_root: Path) -> None:
    with pytest.raises(ValueError, match="engine_ref"):
        p2.load(engine_ref)


@pytest.mark.usefixtures("synth_engine")
def test_backend_env_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("P2_BACKEND", "cuda")
    with pytest.raises(ValueError, match="P2_BACKEND"):
        p2.load(ENGINE_REF)


@pytest.mark.usefixtures("synth_engine")
def test_backend_env_trt_without_tensorrt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("P2_BACKEND", "trt")
    monkeypatch.setitem(sys.modules, "tensorrt", None)
    monkeypatch.delitem(sys.modules, "p2.backends.trt", raising=False)
    with pytest.raises(p2.P2Error, match=r"p2\[trt\]") as error:
        p2.load(ENGINE_REF)
    assert error.value.code == p2.ErrorCode.ENGINE_NOT_FOUND


@pytest.mark.usefixtures("synth_engine")
def test_auto_backend_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "tensorrt", None)
    monkeypatch.delitem(sys.modules, "p2.backends.trt", raising=False)
    with p2.load(ENGINE_REF) as handle:
        assert p2.info(handle)["precision"] == "FP32"


@pytest.mark.usefixtures("synth_engine")
def test_backend_env_ort(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("P2_BACKEND", "ort")
    with p2.load(ENGINE_REF) as handle:
        assert p2.info(handle)["precision"] == "FP32"


def test_import_does_not_load_backends() -> None:
    code = "import sys, p2; print('onnxruntime' in sys.modules, 'tensorrt' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False False"


def test_public_exports() -> None:
    assert set(p2.__all__) == {
        "EngineInfo",
        "ErrorCode",
        "Handle",
        "InferStats",
        "P2Error",
        "__version__",
        "infer",
        "info",
        "load",
        "release",
    }
    assert p2.__version__ == "0.1.0"
