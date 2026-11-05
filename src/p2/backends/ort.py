from collections.abc import Mapping
from pathlib import Path
from typing import Final

import numpy as np
import onnxruntime as ort
from onnxruntime.capi import onnxruntime_pybind11_state as ort_state

from p2.backends.base import TensorArray
from p2.errors import ErrorCode, P2Error
from p2.manifest import Manifest, Precision

CPU_PRECISION: Final = "FP32"
PROVIDERS: Final = ("CPUExecutionProvider",)
ORT_ERRORS: Final = (
    ort_state.EPFail,
    ort_state.EngineError,
    ort_state.Fail,
    ort_state.InvalidArgument,
    ort_state.InvalidGraph,
    ort_state.InvalidProtobuf,
    ort_state.ModelLoaded,
    ort_state.NoModel,
    ort_state.NoSuchFile,
    ort_state.NotFound,
    ort_state.RuntimeException,
)


class OrtSession:
    def __init__(self, session: ort.InferenceSession, output_names: tuple[str, ...]) -> None:
        self._session: ort.InferenceSession | None = session
        self._output_names = output_names

    @property
    def memory_mb(self) -> float:
        return 0.0

    def run(self, inputs: Mapping[str, TensorArray]) -> dict[str, TensorArray]:
        if self._session is None:
            raise P2Error(ErrorCode.RUNTIME_ERROR, "sessão de CPU já encerrada")
        try:
            results = self._session.run(self._output_names, dict(inputs))
        except ORT_ERRORS as err:
            raise P2Error(ErrorCode.RUNTIME_ERROR, f"falha na execução na CPU: {err}") from err
        return {
            name: np.asarray(result)
            for name, result in zip(self._output_names, results, strict=True)
        }

    def close(self) -> None:
        self._session = None


class OrtBackend:
    @property
    def default_precision(self) -> Precision:
        return CPU_PRECISION

    def open(self, manifest: Manifest, directory: Path, precision: Precision) -> OrtSession:
        if precision != CPU_PRECISION:
            raise P2Error(
                ErrorCode.ENGINE_NOT_FOUND,
                f"o backend de CPU só executa {CPU_PRECISION}; {precision} não está disponível",
            )
        onnx_path = directory / manifest["source"]["onnx_file"]
        if not onnx_path.is_file():
            raise P2Error(ErrorCode.ENGINE_NOT_FOUND, f"ONNX ausente: {onnx_path}")
        try:
            session = ort.InferenceSession(str(onnx_path), providers=PROVIDERS)
        except MemoryError as err:
            raise P2Error(
                ErrorCode.OUT_OF_MEMORY, f"memória insuficiente ao carregar {onnx_path}"
            ) from err
        except ORT_ERRORS as err:
            raise P2Error(
                ErrorCode.ENGINE_NOT_FOUND, f"ONNX não pôde ser carregado: {err}"
            ) from err
        _check_io(session, manifest)
        return OrtSession(session, tuple(manifest["outputs"]))


def _check_io(session: ort.InferenceSession, manifest: Manifest) -> None:
    inputs = {arg.name for arg in session.get_inputs()}
    outputs = {arg.name for arg in session.get_outputs()}
    if inputs != set(manifest["inputs"]) or outputs != set(manifest["outputs"]):
        raise P2Error(
            ErrorCode.ENGINE_NOT_FOUND,
            f"ONNX diverge do manifesto: entradas {sorted(inputs)}, saídas {sorted(outputs)}",
        )
