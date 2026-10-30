import sys

import onnx
import pytest

from check_jetson import (
    EXIT_FAILED,
    EXIT_OK,
    CheckResult,
    check_cuda_roundtrip,
    check_numpy_version,
    check_python_version,
    check_tensorrt,
    check_tensorrt_build,
    check_tensorrt_version,
    parse_version,
    run_checks,
    synthetic_onnx,
    version_in_range,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1.26.4", (1, 26, 4)),
        ("10.3.0.30", (10, 3, 0, 30)),
        ("2.0.0rc1", (2, 0, 0)),
        (" 1.21.6 ", (1, 21, 6)),
    ],
)
def test_parse_version(text: str, expected: tuple[int, ...]) -> None:
    assert parse_version(text) == expected


def test_parse_version_without_number() -> None:
    with pytest.raises(ValueError, match="versão sem número"):
        parse_version("dev")


@pytest.mark.parametrize(
    ("version", "inside"),
    [
        ((1, 21, 5), False),
        ((1, 21, 6), True),
        ((1, 26, 4), True),
        ((2,), False),
        ((2, 0, 1), False),
    ],
)
def test_version_in_range(version: tuple[int, ...], inside: bool) -> None:
    assert version_in_range(version, minimum=(1, 21, 6), maximum_exclusive=(2,)) is inside


@pytest.mark.parametrize(("version", "ok"), [((3, 10, 12), True), ((3, 10, 11), False)])
def test_python_version(version: tuple[int, int, int], ok: bool) -> None:
    assert check_python_version(version).ok is ok


@pytest.mark.parametrize(
    ("version", "ok"),
    [("10.3.0", True), ("10.3.0.30", True), ("10.3.01", False), ("10.4.0", False)],
)
def test_tensorrt_version(version: str, ok: bool) -> None:
    assert check_tensorrt_version(version).ok is ok


@pytest.mark.parametrize(("version", "ok"), [("1.21.5", False), ("1.26.4", True), ("2.2.6", False)])
def test_numpy_version(version: str, ok: bool) -> None:
    assert check_numpy_version(version).ok is ok


def test_missing_tensorrt_is_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "tensorrt", None)
    assert not check_tensorrt().ok
    result = check_tensorrt_build()
    assert not result.ok
    assert "import tensorrt falhou" in result.detail


def test_missing_cudart_is_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "cuda.cudart", None)
    result = check_cuda_roundtrip()
    assert not result.ok
    assert "cudart" in result.detail


def test_synthetic_onnx_is_valid() -> None:
    model = onnx.load_from_string(synthetic_onnx())
    onnx.checker.check_model(model)


def test_run_checks_all_ok(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO")
    checks = [lambda: CheckResult("a", True, "x"), lambda: CheckResult("b", True, "y")]
    assert run_checks(checks) == EXIT_OK
    assert "OK a: x" in caplog.text
    assert "placa pronta" in caplog.text


def test_run_checks_continues_after_failure(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO")
    calls: list[str] = []

    def failing() -> CheckResult:
        calls.append("failing")
        return CheckResult("a", False, "quebrou")

    def passing() -> CheckResult:
        calls.append("passing")
        return CheckResult("b", True, "ok")

    assert run_checks([failing, passing]) == EXIT_FAILED
    assert calls == ["failing", "passing"]
    assert "FALHA a: quebrou" in caplog.text
    assert "1 de 2 verificações falharam" in caplog.text
