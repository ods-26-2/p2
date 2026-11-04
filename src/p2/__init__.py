from importlib.metadata import version

from p2.api import infer, info, load, release
from p2.backends.base import TensorArray
from p2.errors import ErrorCode, P2Error
from p2.handle import EngineInfo, Handle, InferStats

__version__ = version("p2")

__all__ = [
    "EngineInfo",
    "ErrorCode",
    "Handle",
    "InferStats",
    "P2Error",
    "TensorArray",
    "__version__",
    "infer",
    "info",
    "load",
    "release",
]
