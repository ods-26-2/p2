import threading
from collections.abc import Mapping
from types import TracebackType
from typing import TypedDict

import numpy as np
from typing_extensions import Self, override

from p2.backends.base import FloatArray, Session
from p2.manifest import Precision


class EngineInfo(TypedDict):
    engine_version: str
    input_shape: dict[str, list[int]]
    output_shape: dict[str, list[int]]
    precision: Precision
    max_batch: int


class InferStats(TypedDict):
    latency_ms: float
    gpu_memory_mb: float


class Handle:
    def __init__(self, *, engine_ref: str, info: EngineInfo, session: Session) -> None:
        self._engine_ref = engine_ref
        self._info = info
        self._session = session
        self._memory_mb = session.memory_mb
        self._lock = threading.Lock()
        self._released = False

    @property
    def engine_ref(self) -> str:
        return self._engine_ref

    @property
    def memory_mb(self) -> float:
        return self._memory_mb

    @property
    def released(self) -> bool:
        return self._released

    def input_shapes(self) -> dict[str, tuple[int, ...]]:
        return {name: tuple(shape) for name, shape in self._info["input_shape"].items()}

    def info(self) -> EngineInfo:
        self.ensure_open()
        return {
            "engine_version": self._info["engine_version"],
            "input_shape": _copy_shapes(self._info["input_shape"]),
            "output_shape": _copy_shapes(self._info["output_shape"]),
            "precision": self._info["precision"],
            "max_batch": self._info["max_batch"],
        }

    def run(self, inputs: Mapping[str, FloatArray]) -> dict[str, FloatArray]:
        with self._lock:
            self.ensure_open()
            outputs = self._session.run(inputs)
            return {name: np.array(array, dtype=np.float32) for name, array in outputs.items()}

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
            self._session.close()

    def ensure_open(self) -> None:
        if self._released:
            raise ValueError(f"handle de '{self._engine_ref}' já foi liberado")

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()

    @override
    def __repr__(self) -> str:
        state = "liberado" if self._released else "aberto"
        return f"Handle({self._info['engine_version']!r}, {state})"


def _copy_shapes(shapes: Mapping[str, list[int]]) -> dict[str, list[int]]:
    return {name: list(shape) for name, shape in shapes.items()}
