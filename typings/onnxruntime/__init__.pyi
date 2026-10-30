from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt

__version__: str

class NodeArg:
    @property
    def name(self) -> str: ...
    @property
    def shape(self) -> list[int | str | None]: ...
    @property
    def type(self) -> str: ...

class SessionOptions:
    log_severity_level: int
    def __init__(self) -> None: ...

class InferenceSession:
    def __init__(
        self,
        path_or_bytes: str | bytes,
        sess_options: SessionOptions | None = None,
        providers: Sequence[str] | None = None,
    ) -> None: ...
    def get_inputs(self) -> list[NodeArg]: ...
    def get_outputs(self) -> list[NodeArg]: ...
    def run(
        self,
        output_names: Sequence[str] | None,
        input_feed: Mapping[str, npt.NDArray[np.generic]],
    ) -> list[npt.NDArray[np.generic]]: ...

def get_available_providers() -> list[str]: ...
