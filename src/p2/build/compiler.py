import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import numpy as np
import tensorrt as trt
from typing_extensions import Buffer

from p2.backends.trt import CollectingLogger, TrtBackend, max_shape
from p2.build.output import WORKSPACE_MB
from p2.build.package import OnnxIO, pin_fixed_axes
from p2.build.platform import read_platform
from p2.errors import BuildError, P2Error
from p2.manifest import Manifest, PlatformInfo, Precision, ShapeProfile, TensorSpec

logger = logging.getLogger(__name__)

BYTES_PER_MB: Final = 1 << 20
NETWORK_FLAGS: Final = 0


class TrtCompiler:
    def __init__(self) -> None:
        self._logger = CollectingLogger()

    def platform(self) -> PlatformInfo:
        return read_platform(trt.__version__)

    def smoke_test(self, directory: Path, manifest: Manifest, precision: Precision) -> None:
        try:
            session = TrtBackend().open(manifest, directory, precision)
        except P2Error as err:
            raise BuildError(f"engine {precision} não abriu para o teste: {err}") from err
        try:
            zeros = {
                name: np.zeros(max_shape(spec), dtype=np.dtype(spec["dtype"]))
                for name, spec in manifest["inputs"].items()
            }
            session.run(zeros)
        except P2Error as err:
            raise BuildError(
                f"infer de teste {precision} com entrada zerada falhou: {err}"
            ) from err
        finally:
            session.close()

    def compile(
        self,
        onnx_path: Path,
        engine_path: Path,
        precision: Precision,
        profiles: Mapping[str, ShapeProfile],
    ) -> OnnxIO:
        try:
            model = pin_fixed_axes(onnx_path.read_bytes(), profiles)
            serialized = self._build(model, precision, profiles)
            engine_io = self._inspect(serialized, precision)
        except (RuntimeError, TypeError) as err:
            raise BuildError(f"TensorRT falhou ao compilar {precision}: {err}") from err
        except MemoryError as err:
            raise BuildError(f"memória insuficiente ao compilar {precision}") from err
        engine_path.write_bytes(serialized)
        return engine_io

    def _build(
        self, model: bytes, precision: Precision, profiles: Mapping[str, ShapeProfile]
    ) -> bytes:
        builder = trt.Builder(self._logger)
        network = builder.create_network(NETWORK_FLAGS)
        parser = trt.OnnxParser(network, self._logger)
        if not parser.parse(model):
            errors = "; ".join(parser.get_error(index).desc() for index in range(parser.num_errors))
            self._logger.take_errors()
            raise BuildError(f"parser ONNX do TensorRT falhou: {errors}")
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_MB * BYTES_PER_MB)
        config.clear_flag(trt.BuilderFlag.TF32)
        if precision == "FP16":
            config.set_flag(trt.BuilderFlag.FP16)
        if profiles:
            profile = builder.create_optimization_profile()
            for name, bounds in profiles.items():
                profile.set_shape(name, bounds["min"], bounds["opt"], bounds["max"])
            config.add_optimization_profile(profile)
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise BuildError(f"compilação {precision} falhou: {self._logger.take_errors()}")
        return bytes(memoryview(serialized))

    def _inspect(self, serialized: Buffer, precision: Precision) -> OnnxIO:
        runtime = trt.Runtime(self._logger)
        engine = runtime.deserialize_cuda_engine(serialized)
        if engine is None:
            raise BuildError(
                f"engine {precision} compilado não pôde ser desserializado: "
                f"{self._logger.take_errors()}"
            )
        return engine_io(engine)


def engine_io(engine: trt.ICudaEngine) -> OnnxIO:
    inputs: dict[str, TensorSpec] = {}
    outputs: dict[str, TensorSpec] = {}
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        spec: TensorSpec = {
            "shape": list(engine.get_tensor_shape(name)),
            "dtype": np.dtype(trt.nptype(engine.get_tensor_dtype(name))).name,
        }
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            inputs[name] = spec
        elif mode == trt.TensorIOMode.OUTPUT:
            outputs[name] = spec
    return OnnxIO(inputs=inputs, outputs=outputs)
