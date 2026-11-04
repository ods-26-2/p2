import logging
import os
import time
from collections.abc import Mapping
from typing import Final, TypeAlias

import numpy as np
import numpy.typing as npt

from p2.backends.base import Backend, FloatArray, TensorArray
from p2.errors import ErrorCode, P2Error
from p2.handle import EngineInfo, Handle, InferStats
from p2.manifest import Manifest, Precision, as_precision, engine_dir, read_manifest

logger = logging.getLogger(__name__)

InferInput: TypeAlias = npt.NDArray[np.generic] | Mapping[str, npt.NDArray[np.generic]]

BACKEND_ENV: Final = "P2_BACKEND"
PRECISION_OPTION: Final = "precision"
MS_PER_SECOND: Final = 1000.0


def load(engine_ref: str, options: Mapping[str, object] | None = None) -> Handle:
    _check_engine_ref(engine_ref)
    requested = _requested_precision(options)
    backend = select_backend(os.environ)
    directory = engine_dir(engine_ref, os.environ)
    manifest = read_manifest(directory)
    if manifest["engine_ref"] != engine_ref:
        raise P2Error(
            ErrorCode.ENGINE_NOT_FOUND,
            f"manifesto em {directory} é de '{manifest['engine_ref']}', não de '{engine_ref}'",
        )
    precision = requested or backend.default_precision
    session = backend.open(manifest, directory, precision)
    return Handle(engine_ref=engine_ref, info=_engine_info(manifest, precision), session=session)


def infer(
    handle: Handle,
    input: InferInput,
) -> tuple[dict[str, TensorArray], InferStats]:
    started = time.perf_counter()
    handle.ensure_open()
    feeds = _normalize_input(input, handle.input_shapes())
    outputs = handle.run(feeds)
    latency_ms = (time.perf_counter() - started) * MS_PER_SECOND
    return outputs, {"latency_ms": latency_ms, "gpu_memory_mb": handle.memory_mb}


def info(handle: Handle) -> EngineInfo:
    return handle.info()


def release(handle: Handle) -> None:
    handle.release()


def select_backend(env: Mapping[str, str]) -> Backend:
    choice = env.get(BACKEND_ENV, "")
    match choice:
        case "":
            return _try_trt_backend() or _ort_backend()
        case "ort":
            return _ort_backend()
        case "trt":
            return _trt_backend()
        case _:
            raise ValueError(f"{BACKEND_ENV} inválido: '{choice}' (use 'trt' ou 'ort')")


def _trt_backend() -> Backend:
    try:
        from p2.backends.trt import TrtBackend  # noqa: PLC0415
    except ImportError as err:
        raise P2Error(
            ErrorCode.ENGINE_NOT_FOUND,
            f"TensorRT ou cuda-python indisponíveis ({err}); na Jetson instale 'p2[trt]'",
        ) from err
    return TrtBackend()


def _try_trt_backend() -> Backend | None:
    try:
        from p2.backends.trt import TrtBackend  # noqa: PLC0415
    except ImportError as err:
        logger.debug("TensorRT indisponível (%s); usando a CPU", err)
        return None
    return TrtBackend()


def _ort_backend() -> Backend:
    try:
        from p2.backends.ort import OrtBackend  # noqa: PLC0415
    except ImportError as err:
        raise P2Error(
            ErrorCode.ENGINE_NOT_FOUND,
            f"onnxruntime não instalado; instale com pip install 'p2[cpu]' ({err})",
        ) from err
    return OrtBackend()


def _check_engine_ref(engine_ref: object) -> None:
    if not isinstance(engine_ref, str) or not engine_ref.strip():
        raise ValueError("engine_ref deve ser texto não vazio")
    if "/" in engine_ref or "\\" in engine_ref or engine_ref in {".", ".."}:
        raise ValueError(f"engine_ref inválido: '{engine_ref}'")


def _requested_precision(options: Mapping[str, object] | None) -> Precision | None:
    if options is None:
        return None
    if not isinstance(options, Mapping):
        raise ValueError("options deve ser um dicionário ou None")
    unknown = sorted(str(key) for key in options if key != PRECISION_OPTION)
    if unknown:
        raise ValueError(f"opção desconhecida: {', '.join(unknown)} (só '{PRECISION_OPTION}')")
    if PRECISION_OPTION not in options:
        return None
    value = options[PRECISION_OPTION]
    precision = as_precision(value)
    if precision is None:
        raise ValueError(f"precisão inválida: {value!r} (use 'FP16' ou 'FP32')")
    return precision


def _engine_info(manifest: Manifest, precision: Precision) -> EngineInfo:
    return {
        "engine_version": f"{manifest['engine_ref']}-{precision.lower()}",
        "input_shape": {name: list(spec["shape"]) for name, spec in manifest["inputs"].items()},
        "output_shape": {name: list(spec["shape"]) for name, spec in manifest["outputs"].items()},
        "precision": precision,
        "max_batch": manifest["max_batch"],
    }


def _normalize_input(value: object, shapes: Mapping[str, tuple[int, ...]]) -> dict[str, FloatArray]:
    if isinstance(value, np.ndarray):
        if len(shapes) != 1:
            raise P2Error(
                ErrorCode.INVALID_INPUT_SHAPE,
                f"o modelo tem {len(shapes)} entradas; passe {{nome: array}} com {sorted(shapes)}",
            )
        (name,) = shapes
        return {name: _check_array(name, value, shapes[name])}
    if not isinstance(value, Mapping):
        raise P2Error(
            ErrorCode.INVALID_INPUT_SHAPE,
            f"entrada deve ser np.ndarray ou dict, obtido {type(value).__name__}",
        )
    names = {str(key) for key in value}
    if names != set(shapes):
        raise P2Error(
            ErrorCode.INVALID_INPUT_SHAPE,
            f"nomes de entrada {sorted(names)} diferentes dos esperados {sorted(shapes)}",
        )
    return {name: _check_array(name, value[name], shape) for name, shape in shapes.items()}


def _check_array(name: str, value: object, shape: tuple[int, ...]) -> FloatArray:
    if not isinstance(value, np.ndarray):
        raise P2Error(
            ErrorCode.INVALID_INPUT_SHAPE,
            f"entrada '{name}' deve ser np.ndarray, obtido {type(value).__name__}",
        )
    if value.dtype != np.float32:
        raise P2Error(
            ErrorCode.INVALID_INPUT_SHAPE,
            f"entrada '{name}' deve ser float32, obtido {value.dtype}",
        )
    if value.shape != shape:
        raise P2Error(
            ErrorCode.INVALID_INPUT_SHAPE,
            f"entrada '{name}' com forma {list(value.shape)}, esperado {list(shape)}",
        )
    return np.ascontiguousarray(value, dtype=np.float32)
