import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

import p2
from helpers.model_package import ENGINE_REF, INPUT_NAME, INPUT_SHAPE, PackageSpec, build_engine
from p2.errors import ErrorCode, P2BaseError, P2Error
from p2.manifest import MANIFEST_FILE


def _zeros(shape: tuple[int, ...] = INPUT_SHAPE) -> npt.NDArray[np.float32]:
    return np.zeros(shape, dtype=np.float32)


def test_error_code_compares_to_string() -> None:
    assert ErrorCode.ENGINE_NOT_FOUND == "ENGINE_NOT_FOUND"
    assert str(ErrorCode.RUNTIME_ERROR) == "RUNTIME_ERROR"
    assert f"{ErrorCode.OUT_OF_MEMORY}" == "OUT_OF_MEMORY"
    assert {code.value for code in ErrorCode} == {
        "ENGINE_NOT_FOUND",
        "INCOMPATIBLE_VERSION",
        "OUT_OF_MEMORY",
        "INVALID_INPUT_SHAPE",
        "RUNTIME_ERROR",
    }


def test_p2_error_format() -> None:
    error = P2Error(ErrorCode.ENGINE_NOT_FOUND, "pasta ausente")
    assert error.code == "ENGINE_NOT_FOUND"
    assert error.detail == "pasta ausente"
    assert str(error) == "ENGINE_NOT_FOUND: pasta ausente"
    assert isinstance(error, P2BaseError)


def _assert_code(error: pytest.ExceptionInfo[P2Error], code: ErrorCode) -> None:
    assert error.value.code == code


@pytest.mark.usefixtures("engine_root")
def test_missing_engine_dir() -> None:
    with pytest.raises(P2Error, match="pasta do engine não existe") as error:
        p2.load(ENGINE_REF)
    _assert_code(error, ErrorCode.ENGINE_NOT_FOUND)


def test_missing_manifest(synth_engine: Path) -> None:
    (synth_engine / MANIFEST_FILE).unlink()
    with pytest.raises(P2Error, match="manifesto ausente") as error:
        p2.load(ENGINE_REF)
    _assert_code(error, ErrorCode.ENGINE_NOT_FOUND)


@pytest.mark.parametrize("content", ["{quebrado", '{"schema_version": 1}', "[]"])
def test_invalid_manifest(synth_engine: Path, content: str) -> None:
    (synth_engine / MANIFEST_FILE).write_text(content, encoding="utf-8")
    with pytest.raises(P2Error, match="manifesto") as error:
        p2.load(ENGINE_REF)
    _assert_code(error, ErrorCode.ENGINE_NOT_FOUND)


def test_manifest_of_other_engine_ref(synth_engine: Path, engine_root: Path) -> None:
    other = engine_root / "outro-1.0.0"
    synth_engine.rename(other)
    with pytest.raises(P2Error, match=r"não de 'outro-1\.0\.0'") as error:
        p2.load("outro-1.0.0")
    _assert_code(error, ErrorCode.ENGINE_NOT_FOUND)


def test_missing_onnx(synth_engine: Path) -> None:
    (synth_engine / "model.onnx").unlink()
    with pytest.raises(P2Error, match="ONNX ausente") as error:
        p2.load(ENGINE_REF)
    _assert_code(error, ErrorCode.ENGINE_NOT_FOUND)


def test_corrupt_onnx(synth_engine: Path) -> None:
    (synth_engine / "model.onnx").write_bytes(b"garbage")
    with pytest.raises(P2Error, match="ONNX não pôde ser carregado") as error:
        p2.load(ENGINE_REF)
    _assert_code(error, ErrorCode.ENGINE_NOT_FOUND)


def test_onnx_diverging_from_manifest(synth_engine: Path) -> None:
    manifest_path = synth_engine / MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["inputs"] = {"outro": manifest["inputs"][INPUT_NAME]}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(P2Error, match="diverge do manifesto") as error:
        p2.load(ENGINE_REF)
    _assert_code(error, ErrorCode.ENGINE_NOT_FOUND)


@pytest.mark.usefixtures("synth_engine")
def test_fp16_on_cpu_is_engine_not_found() -> None:
    with pytest.raises(P2Error, match="só executa FP32") as error:
        p2.load(ENGINE_REF, {"precision": "FP16"})
    _assert_code(error, ErrorCode.ENGINE_NOT_FOUND)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (np.zeros((1, 3, 8, 7), dtype=np.float32), "forma"),
        (np.zeros((3, 8, 8), dtype=np.float32), "forma"),
        (np.zeros(INPUT_SHAPE, dtype=np.float64), "float32"),
        (np.zeros(INPUT_SHAPE, dtype=np.float16), "float32"),
        ({"outro": np.zeros(INPUT_SHAPE, dtype=np.float32)}, "nomes de entrada"),
        (
            {INPUT_NAME: np.zeros(INPUT_SHAPE, dtype=np.float32), "extra": np.zeros(1)},
            "nomes de entrada",
        ),
        ({INPUT_NAME: [[0.0]]}, "deve ser np.ndarray"),
        ([0.0], "np.ndarray ou dict"),
    ],
)
@pytest.mark.usefixtures("synth_engine")
def test_invalid_input(value: object, message: str) -> None:
    with p2.load(ENGINE_REF) as handle:
        with pytest.raises(P2Error, match=message) as error:
            p2.infer(handle, value)  # type: ignore[arg-type]
        _assert_code(error, ErrorCode.INVALID_INPUT_SHAPE)


def test_runtime_error(engine_root: Path, tmp_path: Path) -> None:
    build_engine(tmp_path / "pkg", engine_root, PackageSpec(fails_at_runtime=True))
    with p2.load(ENGINE_REF) as handle:
        with pytest.raises(P2Error, match="falha na execução") as error:
            p2.infer(handle, _zeros())
        _assert_code(error, ErrorCode.RUNTIME_ERROR)
