import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import tensorrt as trt
from cuda import cudart
from typing_extensions import override

from p2.backends.base import TensorArray
from p2.backends.variants import memory_mb, read_engine, select_variant
from p2.errors import ErrorCode, P2Error
from p2.manifest import DYNAMIC_DIM, Manifest, Precision, TensorSpec, is_dynamic

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
class DeviceInput:
    name: str
    pointer: int
    nbytes: int
    dynamic: bool


@dataclass(frozen=True, slots=True)
class DeviceOutput:
    name: str
    pointer: int
    nbytes: int
    shape: tuple[int, ...] | None
    host: TensorArray


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
        inputs: tuple[DeviceInput, ...],
        outputs: tuple[DeviceOutput, ...],
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

    def run(self, inputs: Mapping[str, TensorArray]) -> dict[str, TensorArray]:
        if self._parts is None:
            raise P2Error(ErrorCode.RUNTIME_ERROR, "sessão TensorRT já encerrada")
        context = self._parts.context
        for tensor in self._inputs:
            self._send(context, tensor, inputs[tensor.name])
        shapes = {tensor.name: _output_shape(context, tensor) for tensor in self._outputs}
        if not context.execute_async_v3(int(self._stream)):
            errors = self._parts.trt_logger.take_errors()
            raise P2Error(ErrorCode.RUNTIME_ERROR, f"execute_async_v3 falhou: {errors}")
        views = {
            tensor.name: _host_view(tensor.host, shapes[tensor.name]) for tensor in self._outputs
        }
        for output in self._outputs:
            view = views[output.name]
            _check_run(
                cudart.cudaMemcpyAsync(
                    view.ctypes.data, output.pointer, view.nbytes, DEVICE_TO_HOST, self._stream
                ),
                f"cópia da saída '{output.name}' para a CPU",
            )
        _check_run(cudart.cudaStreamSynchronize(self._stream), "cudaStreamSynchronize")
        return views

    def close(self) -> None:
        if self._parts is None:
            return
        _free(tensor.pointer for tensor in self._inputs)
        _free(tensor.pointer for tensor in self._outputs)
        cudart.cudaStreamDestroy(self._stream)
        self._parts = None

    def _send(
        self, context: trt.IExecutionContext, tensor: DeviceInput, source: TensorArray
    ) -> None:
        if tensor.dynamic and not context.set_input_shape(tensor.name, source.shape):
            raise P2Error(
                ErrorCode.RUNTIME_ERROR,
                f"TensorRT recusou a forma {list(source.shape)} de '{tensor.name}'",
            )
        _check_run(
            cudart.cudaMemcpyAsync(
                tensor.pointer, source.ctypes.data, source.nbytes, HOST_TO_DEVICE, self._stream
            ),
            f"cópia da entrada '{tensor.name}' para a GPU",
        )


class TrtBackend:
    @property
    def default_precision(self) -> Precision:
        return DEFAULT_PRECISION

    def open(self, manifest: Manifest, directory: Path, precision: Precision) -> TrtSession:
        variant = select_variant(manifest, precision, trt.__version__)
        data = read_engine(directory, variant)
        parts = _deserialize(data, variant["file"])
        inputs = _allocate_inputs(manifest["inputs"])
        try:
            _set_max_shapes(parts.context, manifest["inputs"])
            outputs = _allocate_outputs(parts.context, manifest["outputs"])
        except P2Error:
            _free(tensor.pointer for tensor in inputs)
            raise
        pointers = [(tensor.name, tensor.pointer) for tensor in inputs]
        pointers += [(tensor.name, tensor.pointer) for tensor in outputs]
        try:
            stream = _create_stream()
            _bind(parts.context, pointers)
        except P2Error:
            _free(pointer for _, pointer in pointers)
            raise
        total = memory_mb(
            engine_bytes=len(data),
            device_memory=parts.engine.device_memory_size_v2,
            buffer_bytes=sum(tensor.nbytes for tensor in inputs)
            + sum(tensor.nbytes for tensor in outputs),
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


def max_shape(spec: TensorSpec) -> list[int]:
    profile = spec.get("profile")
    return spec["shape"] if profile is None else profile["max"]


def _allocate_inputs(specs: Mapping[str, TensorSpec]) -> tuple[DeviceInput, ...]:
    tensors: list[DeviceInput] = []
    for name, spec in specs.items():
        nbytes = _nbytes(max_shape(spec), spec["dtype"])
        try:
            pointer = _malloc(name, nbytes)
        except P2Error:
            _free(tensor.pointer for tensor in tensors)
            raise
        dynamic = "profile" in spec
        tensors.append(DeviceInput(name=name, pointer=pointer, nbytes=nbytes, dynamic=dynamic))
    return tuple(tensors)


def _set_max_shapes(context: trt.IExecutionContext, specs: Mapping[str, TensorSpec]) -> None:
    for name, spec in specs.items():
        if "profile" in spec and not context.set_input_shape(name, max_shape(spec)):
            raise P2Error(
                ErrorCode.ENGINE_NOT_FOUND,
                f"engine não aceita a forma máxima {max_shape(spec)} de '{name}' do manifesto",
            )


def _allocate_outputs(
    context: trt.IExecutionContext, specs: Mapping[str, TensorSpec]
) -> tuple[DeviceOutput, ...]:
    tensors: list[DeviceOutput] = []
    for name, spec in specs.items():
        try:
            capacity = _capacity_shape(context, name, spec["shape"])
            host = np.empty(int(np.prod(capacity)), dtype=np.dtype(spec["dtype"]))
            pointer = _malloc(name, host.nbytes)
        except P2Error:
            _free(tensor.pointer for tensor in tensors)
            raise
        shape = None if is_dynamic(spec["shape"]) else tuple(spec["shape"])
        tensors.append(
            DeviceOutput(name=name, pointer=pointer, nbytes=host.nbytes, shape=shape, host=host)
        )
    return tuple(tensors)


def _capacity_shape(
    context: trt.IExecutionContext, name: str, declared: Sequence[int]
) -> list[int]:
    if not is_dynamic(declared):
        return list(declared)
    shape = list(context.get_tensor_shape(name))
    if DYNAMIC_DIM in shape:
        raise P2Error(
            ErrorCode.ENGINE_NOT_FOUND,
            f"saída '{name}' tem forma dependente dos dados ({shape}); não suportado",
        )
    return shape


def _output_shape(context: trt.IExecutionContext, tensor: DeviceOutput) -> tuple[int, ...]:
    if tensor.shape is not None:
        return tensor.shape
    shape = tuple(context.get_tensor_shape(tensor.name))
    if DYNAMIC_DIM in shape:
        raise P2Error(
            ErrorCode.RUNTIME_ERROR, f"forma da saída '{tensor.name}' indefinida: {list(shape)}"
        )
    return shape


def _host_view(host: TensorArray, shape: tuple[int, ...]) -> TensorArray:
    count = int(np.prod(shape))
    return host[:count].reshape(shape)


def _nbytes(shape: Sequence[int], dtype: str) -> int:
    return int(np.prod(shape)) * np.dtype(dtype).itemsize


def _malloc(name: str, nbytes: int) -> int:
    error, pointer = cudart.cudaMalloc(nbytes)
    if error != SUCCESS:
        raise P2Error(
            ErrorCode.OUT_OF_MEMORY,
            f"cudaMalloc de {nbytes} bytes para '{name}' falhou: {error.name}",
        )
    return pointer


def _create_stream() -> cudart.cudaStream_t:
    error, stream = cudart.cudaStreamCreate()
    if error != SUCCESS:
        raise P2Error(ErrorCode.OUT_OF_MEMORY, f"cudaStreamCreate falhou: {error.name}")
    return stream


def _bind(context: trt.IExecutionContext, pointers: Iterable[tuple[str, int]]) -> None:
    for name, pointer in pointers:
        if not context.set_tensor_address(name, pointer):
            raise P2Error(
                ErrorCode.ENGINE_NOT_FOUND,
                f"engine não tem o tensor '{name}' declarado no manifesto",
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
