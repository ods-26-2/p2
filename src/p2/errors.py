from enum import Enum

from typing_extensions import override


class P2BaseError(Exception):
    pass


class BuildError(P2BaseError):
    pass


class ErrorCode(str, Enum):
    ENGINE_NOT_FOUND = "ENGINE_NOT_FOUND"
    INCOMPATIBLE_VERSION = "INCOMPATIBLE_VERSION"
    OUT_OF_MEMORY = "OUT_OF_MEMORY"
    INVALID_INPUT_SHAPE = "INVALID_INPUT_SHAPE"
    RUNTIME_ERROR = "RUNTIME_ERROR"

    @override
    def __str__(self) -> str:
        return self.value


class P2Error(P2BaseError):
    code: ErrorCode
    detail: str

    def __init__(self, code: ErrorCode, detail: str) -> None:
        super().__init__(f"{code.value}: {detail}")
        self.code = code
        self.detail = detail
