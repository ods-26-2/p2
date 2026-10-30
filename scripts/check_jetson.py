#!/usr/bin/env python3
import logging
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final, TypeAlias

import numpy as np

logger = logging.getLogger(__name__)

EXPECTED_PYTHON: Final = (3, 10, 12)
EXPECTED_TENSORRT: Final = "10.3.0"
NUMPY_MINIMUM: Final = (1, 21, 6)
NUMPY_MAXIMUM_EXCLUSIVE: Final = (2,)
WORKSPACE_BYTES: Final = 1 << 30
PROBE_ELEMENTS: Final = 1024
PROBE_SHAPE: Final = (1, 3, 8, 8)
ONNX_OPSET: Final = 17
ONNX_IR_VERSION: Final = 10
EXIT_OK: Final = 0
EXIT_FAILED: Final = 1
VERSION_PREFIX: Final = re.compile(r"^\d+(?:\.\d+)*")


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


Check: TypeAlias = Callable[[], CheckResult]


def parse_version(text: str) -> tuple[int, ...]:
    match = VERSION_PREFIX.match(text.strip())
    if match is None:
        raise ValueError(f"versão sem número: {text!r}")
    return tuple(int(part) for part in match.group().split("."))


def version_in_range(
    version: tuple[int, ...],
    *,
    minimum: tuple[int, ...],
    maximum_exclusive: tuple[int, ...],
) -> bool:
    return minimum <= version < maximum_exclusive


def check_python_version(version: Sequence[int]) -> CheckResult:
    found = tuple(version[:3])
    expected = ".".join(str(part) for part in EXPECTED_PYTHON)
    detail = f"esperado {expected}, encontrado {'.'.join(str(part) for part in found)}"
    return CheckResult("Python", found == EXPECTED_PYTHON, detail)


def check_tensorrt_version(version: str) -> CheckResult:
    matches = version == EXPECTED_TENSORRT or version.startswith(f"{EXPECTED_TENSORRT}.")
    return CheckResult("TensorRT", matches, f"esperado {EXPECTED_TENSORRT}, encontrado {version}")


def check_numpy_version(version: str) -> CheckResult:
    ok = version_in_range(
        parse_version(version),
        minimum=NUMPY_MINIMUM,
        maximum_exclusive=NUMPY_MAXIMUM_EXCLUSIVE,
    )
    return CheckResult("numpy", ok, f"esperado >=1.21.6,<2, encontrado {version}")


def check_python() -> CheckResult:
    return check_python_version(sys.version_info[:3])


def check_tensorrt() -> CheckResult:
    try:
        import tensorrt  # noqa: PLC0415
    except ImportError as err:
        return CheckResult("TensorRT", False, f"import tensorrt falhou: {err}")
    return check_tensorrt_version(tensorrt.__version__)


def check_numpy() -> CheckResult:
    return check_numpy_version(np.__version__)


def check_cuda_roundtrip() -> CheckResult:
    name = "cuda-python"
    try:
        from cuda import cudart  # noqa: PLC0415
    except ImportError as err:
        return CheckResult(name, False, f"from cuda import cudart falhou: {err}")
    host = np.arange(PROBE_ELEMENTS, dtype=np.float32)
    back = np.zeros_like(host)
    error, device = cudart.cudaMalloc(host.nbytes)
    if error != cudart.cudaError_t.cudaSuccess:
        return CheckResult(name, False, f"cudaMalloc falhou: {error.name}")
    try:
        (to_device,) = cudart.cudaMemcpy(
            device, host.ctypes.data, host.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        )
        (to_host,) = cudart.cudaMemcpy(
            back.ctypes.data, device, host.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
        )
    finally:
        (freed,) = cudart.cudaFree(device)
    failures = [
        f"{label}: {code.name}"
        for label, code in (("cudaMemcpy H→D", to_device), ("cudaMemcpy D→H", to_host))
        if code != cudart.cudaError_t.cudaSuccess
    ]
    if freed != cudart.cudaError_t.cudaSuccess:
        failures.append(f"cudaFree: {freed.name}")
    if not failures and not np.array_equal(host, back):
        failures.append("dados diferentes após ida e volta")
    if failures:
        return CheckResult(name, False, "; ".join(failures))
    return CheckResult(name, True, f"cudaMalloc/cudaMemcpy/cudaFree ok com {host.nbytes} bytes")


def check_tensorrt_build() -> CheckResult:
    name = "Compilação TensorRT"
    try:
        import tensorrt  # noqa: F401, PLC0415
    except ImportError as err:
        return CheckResult(name, False, f"import tensorrt falhou: {err}")
    try:
        model = synthetic_onnx()
    except ImportError as err:
        return CheckResult(name, False, f"import onnx falhou: {err}")
    try:
        return _build_engine(model)
    except RuntimeError as err:
        return CheckResult(name, False, f"TensorRT lançou erro: {err}")


def _build_engine(model: bytes) -> CheckResult:
    import tensorrt as trt  # noqa: PLC0415

    name = "Compilação TensorRT"
    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, trt_logger)
    if not parser.parse(model):
        errors = "; ".join(parser.get_error(index).desc() for index in range(parser.num_errors))
        return CheckResult(name, False, f"parser ONNX falhou: {errors}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_BYTES)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        return CheckResult(name, False, "build_serialized_network devolveu None")
    return CheckResult(name, True, f"engine de {serialized.nbytes} bytes")


def synthetic_onnx() -> bytes:
    from onnx import TensorProto, helper  # noqa: PLC0415

    shape = list(PROBE_SHAPE)
    graph = helper.make_graph(
        [helper.make_node("Relu", ["x"], ["y"])],
        "check_jetson",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", ONNX_OPSET)],
        ir_version=ONNX_IR_VERSION,
    )
    return model.SerializeToString()


ALL_CHECKS: Final[tuple[Check, ...]] = (
    check_python,
    check_tensorrt,
    check_numpy,
    check_cuda_roundtrip,
    check_tensorrt_build,
)


def run_checks(checks: Sequence[Check]) -> int:
    results: list[CheckResult] = []
    for check in checks:
        result = check()
        results.append(result)
        level = logging.INFO if result.ok else logging.ERROR
        status = "OK" if result.ok else "FALHA"
        logger.log(level, "%s %s: %s", status, result.name, result.detail)
    failed = sum(1 for result in results if not result.ok)
    if failed:
        logger.error("%d de %d verificações falharam", failed, len(results))
        return EXIT_FAILED
    logger.info("placa pronta: %d verificações ok", len(results))
    return EXIT_OK


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    return run_checks(ALL_CHECKS)


if __name__ == "__main__":
    raise SystemExit(main())
