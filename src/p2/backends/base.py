from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, TypeAlias

import numpy as np
import numpy.typing as npt

from p2.manifest import Manifest, Precision

FloatArray: TypeAlias = npt.NDArray[np.float32]
TensorArray: TypeAlias = npt.NDArray[np.generic]


class Session(Protocol):
    @property
    def memory_mb(self) -> float: ...

    def run(self, inputs: Mapping[str, TensorArray]) -> Mapping[str, TensorArray]: ...

    def close(self) -> None: ...


class Backend(Protocol):
    @property
    def default_precision(self) -> Precision: ...

    def open(self, manifest: Manifest, directory: Path, precision: Precision) -> Session: ...
