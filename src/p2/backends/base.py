from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, TypeAlias

import numpy as np
import numpy.typing as npt

from p2.manifest import Manifest, Precision

FloatArray: TypeAlias = npt.NDArray[np.float32]


class Session(Protocol):
    @property
    def memory_mb(self) -> float: ...

    def run(self, inputs: Mapping[str, FloatArray]) -> Mapping[str, FloatArray]: ...

    def close(self) -> None: ...


class Backend(Protocol):
    @property
    def default_precision(self) -> Precision: ...

    def open(self, manifest: Manifest, directory: Path, precision: Precision) -> Session: ...
