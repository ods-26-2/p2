import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import tensorrt as trt
from cuda import cudart
from typing_extensions import override

from p2.backends.base import FloatArray
from p2.backends.variants import memory_mb, read_engine, select_variant
from p2.errors import ErrorCode, P2Error
from p2.manifest import Manifest, Precision, TensorSpec

logger = logging.getLogger(__name__)

DEFAULT_PRECISION: Final = "FP16"
SUCCESS: Final = cudart.cudaError_t.cudaSuccess
HOST_TO_DEVICE: Final = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
DEVICE_TO_HOST: Final = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost


class CollectingLogger(trt.ILogger):
    def __init__(self) -> None:
        trt.ILogger.__init__(self)
        self.errors: list[str] = []

    @override
    def log(self, severity: trt.ILogger.Severity, msg: str) -> None:
        if severity in (trt.ILogger.INTERNAL_ERROR, trt.ILogger.ERROR):
            self.errors.append(msg)
            logger.error("TensorRT: %s", msg)
        elif severity == trt.ILogger.WARNING:
            logger.warning("TensorRT: %s", msg)
        else:
            logger.debug("TensorRT: %s", msg)

    def take_errors(self) -> str:
        message = "; ".join(self.errors) or "sem detalhe do TensorRT"
        self.errors.clear()
        return message


@dataclass(frozen=True, slots=True)
class DeviceTensor:
    name: str
    pointer: int
    nbytes: int
    host: FloatArray


@dataclass(frozen=True, slots=True)
class EngineParts:
    trt_logger: CollectingLogger
    runtime: trt.Runtime
    engine: trt.ICudaEngine
    context: trt.IExecutionContext


class TrtSession:
    def __init__(
        self,
        *,
        parts: EngineParts,
        stream: cudart.cudaStream_t,
        inputs: tuple[DeviceTensor, ...],
        outputs: tuple[DeviceTensor, ...],
        memory_mb: float,
    ) -> None:
        self._parts: EngineParts | None = parts
        self._stream = stream
        self._inputs = inputs
        self._outputs = outputs
        self._memory_mb = memory_mb

    @property
    def memory_mb(self) -> float:
        return self._memory_mb

    def run(self, inputs: Mapping[str, FloatArray]) -> dict[str, FloatArray]:
        if self._parts is None:
            raise P2Error(ErrorCode.RUNTIME_ERROR, "sessão TensorRT já encerrada")
        for tensor in self._inputs:
            source = inputs[tensor.name]
            _check_run(
                cudart.cudaMemcpyAsync(
                    tensor.pointer, source.ctypes.data, tensor.nbytes, HOST_TO_DEVICE, self._stream
                ),
                f"cópia da entrada '{tensor.name}' para a GPU",
            )
        if not self._parts.context.execute_async_v3(int(self._stream)):
            errors = self._parts.trt_logger.take_errors()
            raise P2Error(ErrorCode.RUNTIME_ERROR, f"execute_async_v3 falhou: {errors}")
        for tensor in self._outputs:
            _check_run(
                cudart.cudaMemcpyAsync(
                    tensor.host.ctypes.data,
                    tensor.pointer,
                    tensor.nbytes,
                    DEVICE_TO_HOST,
                    self._stream,
                ),
                f"cópia da saída '{tensor.name}' para a CPU",
            )
        _check_run(cudart.cudaStreamSynchronize(self._stream), "cudaStreamSynchronize")
        return {tensor.name: tensor.host for tensor in self._outputs}

    def close(self) -> None:
        if self._parts is None:
            return
        _free(tensor.pointer for tensor in (*self._inputs, *self._outputs))
        cudart.cudaStreamDestroy(self._stream)
        self._parts = None


class TrtBackend:
    @property
    def default_precision(self) -> Precision:
        return DEFAULT_PRECISION

    def open(self, manifest: Manifest, directory: Path, precision: Precision) -> TrtSession:
        variant = select_variant(manifest, precision, trt.__version__)
        data = read_engine(directory, variant)
        parts = _deserialize(data, variant["file"])
        inputs = _allocate(manifest["inputs"])
        try:
            outputs = _allocate(manifest["outputs"])
        except P2Error:
            _free(tensor.pointer for tensor in inputs)
            raise
        tensors = (*inputs, *outputs)
        try:
            stream = _create_stream()
            _bind(parts.context, tensors)
        except P2Error:
            _free(tensor.pointer for tensor in tensors)
            raise
        total = memory_mb(
            engine_bytes=len(data),
            device_memory=parts.engine.device_memory_size_v2,
            buffer_bytes=sum(tensor.nbytes for tensor in tensors),
        )
        return TrtSession(
            parts=parts, stream=stream, inputs=inputs, outputs=outputs, memory_mb=total
        )


def _deserialize(data: bytes, name: str) -> EngineParts:
    trt_logger = CollectingLogger()
    runtime = trt.Runtime(trt_logger)
    engine = runtime.deserialize_cuda_engine(data)
    if engine is None:
        raise P2Error(
            ErrorCode.OUT_OF_MEMORY,
            f"falha ao desserializar {name}: {trt_logger.take_errors()}",
        )
    context = engine.create_execution_context()
    if context is None:
        raise P2Error(
            ErrorCode.OUT_OF_MEMORY,
            f"falha ao criar o contexto de execução de {name}: {trt_logger.take_errors()}",
        )
    return EngineParts(trt_logger=trt_logger, runtime=runtime, engine=engine, context=context)


def _allocate(specs: Mapping[str, TensorSpec]) -> tuple[DeviceTensor, ...]:
    tensors: list[DeviceTensor] = []
    for name, spec in specs.items():
        host = np.empty(spec["shape"], dtype=np.float32)
        error, pointer = cudart.cudaMalloc(host.nbytes)
        if error != SUCCESS:
            _free(tensor.pointer for tensor in tensors)
            raise P2Error(
                ErrorCode.OUT_OF_MEMORY,
                f"cudaMalloc de {host.nbytes} bytes para '{name}' falhou: {error.name}",
            )
        tensors.append(DeviceTensor(name=name, pointer=pointer, nbytes=host.nbytes, host=host))
    return tuple(tensors)


def _create_stream() -> cudart.cudaStream_t:
    error, stream = cudart.cudaStreamCreate()
    if error != SUCCESS:
        raise P2Error(ErrorCode.OUT_OF_MEMORY, f"cudaStreamCreate falhou: {error.name}")
    return stream


def _bind(context: trt.IExecutionContext, tensors: Iterable[DeviceTensor]) -> None:
    for tensor in tensors:
        if not context.set_tensor_address(tensor.name, tensor.pointer):
            raise P2Error(
                ErrorCode.ENGINE_NOT_FOUND,
                f"engine não tem o tensor '{tensor.name}' declarado no manifesto",
            )


def _free(pointers: Iterable[int]) -> None:
    for pointer in pointers:
        (error,) = cudart.cudaFree(pointer)
        if error != SUCCESS:
            logger.warning("cudaFree falhou: %s", error.name)


def _check_run(result: tuple[cudart.cudaError_t], what: str) -> None:
    (error,) = result
    if error != SUCCESS:
        raise P2Error(ErrorCode.RUNTIME_ERROR, f"{what} falhou: {error.name}")
