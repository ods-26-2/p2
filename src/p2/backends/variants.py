import hashlib
from pathlib import Path
from typing import Final

from p2.build.platform import short_tensorrt_version
from p2.errors import ErrorCode, P2Error
from p2.manifest import Manifest, Precision, VariantInfo

BYTES_PER_MB: Final = 1024 * 1024


def select_variant(
    manifest: Manifest, precision: Precision, installed_tensorrt: str
) -> VariantInfo:
    engine_ref = manifest["engine_ref"]
    if not manifest["variants"]:
        raise P2Error(
            ErrorCode.ENGINE_NOT_FOUND,
            f"pasta de '{engine_ref}' não tem engines TensorRT (gerada com --cpu-only?); "
            "rode p2-build na Jetson ou use P2_BACKEND=ort",
        )
    variant = manifest["variants"].get(precision)
    if variant is None:
        available = ", ".join(sorted(manifest["variants"]))
        raise P2Error(
            ErrorCode.ENGINE_NOT_FOUND,
            f"variante {precision} de '{engine_ref}' não compilada; disponíveis: {available}",
        )
    built = manifest["platform"].get("tensorrt")
    installed = short_tensorrt_version(installed_tensorrt)
    if built != installed:
        raise P2Error(
            ErrorCode.INCOMPATIBLE_VERSION,
            f"'{engine_ref}' foi compilado com TensorRT {built}, instalado {installed}; "
            "recompile com p2-build --force",
        )
    return variant


def read_engine(directory: Path, variant: VariantInfo) -> bytes:
    path = directory / variant["file"]
    try:
        data = path.read_bytes()
    except FileNotFoundError as err:
        raise P2Error(ErrorCode.ENGINE_NOT_FOUND, f"engine ausente: {path}") from err
    actual = hashlib.sha256(data).hexdigest()
    if actual != variant["sha256"]:
        raise P2Error(
            ErrorCode.ENGINE_NOT_FOUND,
            f"engine corrompido: {path} (sha256 {actual}, manifesto {variant['sha256']})",
        )
    return data


def memory_mb(*, engine_bytes: int, device_memory: int, buffer_bytes: int) -> float:
    return (engine_bytes + device_memory + buffer_bytes) / BYTES_PER_MB
