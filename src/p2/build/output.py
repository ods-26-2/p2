import logging
import shutil
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final, Protocol

from p2.build.package import ModelPackage, OnnxIO, sha256_file
from p2.errors import BuildError
from p2.manifest import (
    SCHEMA_VERSION,
    Manifest,
    PlatformInfo,
    Precision,
    VariantInfo,
    write_manifest,
)

logger = logging.getLogger(__name__)

TOOL_NAME: Final = "p2-build"
MAX_BATCH: Final = 1
WORKSPACE_MB: Final = 1024
ENGINE_FILES: Final[Mapping[Precision, str]] = {"FP16": "fp16.engine", "FP32": "fp32.engine"}


class EngineCompiler(Protocol):
    def compile(self, onnx_path: Path, engine_path: Path, precision: Precision) -> OnnxIO: ...

    def platform(self) -> PlatformInfo: ...

    def smoke_test(self, directory: Path, manifest: Manifest, precision: Precision) -> None: ...


@dataclass(frozen=True, slots=True)
class BuildStamp:
    built_at: datetime
    tool_version: str


def build_manifest(
    package: ModelPackage,
    io: OnnxIO,
    *,
    variants: Mapping[Precision, VariantInfo],
    platform: PlatformInfo,
    stamp: BuildStamp,
) -> Manifest:
    return {
        "schema_version": SCHEMA_VERSION,
        "engine_ref": package.engine_ref,
        "source": {
            "onnx_file": package.onnx_path.name,
            "sha256": package.sha256,
            "metadata_file": package.metadata_path.name,
        },
        "inputs": io.inputs,
        "outputs": io.outputs,
        "max_batch": MAX_BATCH,
        "variants": dict(variants),
        "platform": platform,
        "build": {
            "tool": f"{TOOL_NAME} {stamp.tool_version}",
            "built_at": stamp.built_at.isoformat(timespec="seconds"),
            "tf32": False,
            "workspace_mb": WORKSPACE_MB,
        },
    }


def write_cpu_only(
    package: ModelPackage,
    io: OnnxIO,
    *,
    engine_root: Path,
    force: bool,
    stamp: BuildStamp,
) -> Path:
    with staged_engine_dir(engine_root, package.engine_ref, force=force) as staging:
        copy_sources(package, staging)
        manifest = build_manifest(
            package,
            io,
            variants={},
            platform={},
            stamp=stamp,
        )
        write_manifest(manifest, staging)
    return engine_root / package.engine_ref


def write_compiled(  # noqa: PLR0913
    package: ModelPackage,
    io: OnnxIO,
    *,
    engine_root: Path,
    force: bool,
    stamp: BuildStamp,
    precisions: Sequence[Precision],
    compiler: EngineCompiler,
) -> Path:
    platform = compiler.platform()
    with staged_engine_dir(engine_root, package.engine_ref, force=force) as staging:
        copy_sources(package, staging)
        variants = {
            precision: _compile_variant(package, io, staging, precision, compiler)
            for precision in precisions
        }
        manifest = build_manifest(
            package,
            io,
            variants=variants,
            platform=platform,
            stamp=stamp,
        )
        write_manifest(manifest, staging)
        for precision in precisions:
            logger.info("testando %s em %s com entrada zerada", package.engine_ref, precision)
            compiler.smoke_test(staging, manifest, precision)
    return engine_root / package.engine_ref


def _compile_variant(
    package: ModelPackage,
    io: OnnxIO,
    staging: Path,
    precision: Precision,
    compiler: EngineCompiler,
) -> VariantInfo:
    engine_path = staging / ENGINE_FILES[precision]
    logger.info("compilando %s em %s", package.engine_ref, precision)
    engine_io = compiler.compile(staging / package.onnx_path.name, engine_path, precision)
    if engine_io != io:
        raise BuildError(
            f"I/O do engine {precision} diverge do ONNX: "
            f"engine {engine_io.inputs} → {engine_io.outputs}, ONNX {io.inputs} → {io.outputs}"
        )
    return {
        "file": engine_path.name,
        "engine_version": f"{package.engine_ref}-{precision.lower()}",
        "sha256": sha256_file(engine_path),
    }


def copy_sources(package: ModelPackage, directory: Path) -> None:
    shutil.copy2(package.onnx_path, directory / package.onnx_path.name)
    shutil.copy2(package.metadata_path, directory / package.metadata_path.name)


@contextmanager
def staged_engine_dir(engine_root: Path, engine_ref: str, *, force: bool) -> Iterator[Path]:
    target = engine_root / engine_ref
    _ensure_replaceable(target, force=force)
    staging = _create_staging(engine_root, engine_ref)
    try:
        yield staging
        _commit(staging, target, force=force)
    except OSError as err:
        raise BuildError(f"falha ao gravar a pasta do engine em {engine_root}: {err}") from err
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    logger.info("pasta do engine gravada em %s", target)


def _ensure_replaceable(target: Path, *, force: bool) -> None:
    if target.exists() and not force:
        raise BuildError(f"pasta já existe: {target}; use --force para substituir")


def _create_staging(engine_root: Path, engine_ref: str) -> Path:
    try:
        engine_root.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix=f".{engine_ref}.tmp-", dir=engine_root))
    except OSError as err:
        raise BuildError(f"não foi possível criar a pasta em {engine_root}: {err}") from err


def _commit(staging: Path, target: Path, *, force: bool) -> None:
    _ensure_replaceable(target, force=force)
    if not target.exists():
        staging.rename(target)
        return
    backup = Path(tempfile.mkdtemp(prefix=f".{target.name}.old-", dir=target.parent))
    backup.rmdir()
    target.rename(backup)
    try:
        staging.rename(target)
    except OSError:
        backup.rename(target)
        raise
    shutil.rmtree(backup, ignore_errors=True)
    logger.info("pasta anterior substituída: %s", target)
