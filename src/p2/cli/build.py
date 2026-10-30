import argparse
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Final

from p2.build.output import BuildStamp, EngineCompiler, write_compiled, write_cpu_only
from p2.build.package import read_package, validate_package
from p2.errors import BuildError
from p2.manifest import Precision, resolve_engine_root

logger = logging.getLogger(__name__)

PROG: Final = "p2-build"
DISTRIBUTION: Final = "p2"
ALL_PRECISIONS: Final[tuple[Precision, ...]] = ("FP16", "FP32")
EXIT_OK: Final = 0
EXIT_BUILD_ERROR: Final = 1


@dataclass(frozen=True, slots=True)
class BuildArgs:
    package: Path
    precisions: tuple[Precision, ...]
    cpu_only: bool
    force: bool
    engine_dir: Path | None


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args(argv)
    try:
        run(args)
    except BuildError as err:
        logger.error("%s", err)
        return EXIT_BUILD_ERROR
    return EXIT_OK


def run(args: BuildArgs) -> Path:
    package = read_package(args.package)
    io = validate_package(package)
    engine_root = resolve_engine_root(args.engine_dir, os.environ)
    stamp = BuildStamp(built_at=datetime.now(timezone.utc), tool_version=version(DISTRIBUTION))
    if args.cpu_only:
        return write_cpu_only(package, io, engine_root=engine_root, force=args.force, stamp=stamp)
    return write_compiled(
        package,
        io,
        engine_root=engine_root,
        force=args.force,
        stamp=stamp,
        precisions=args.precisions,
        compiler=_trt_compiler(),
    )


def _trt_compiler() -> EngineCompiler:
    try:
        from p2.build.compiler import TrtCompiler  # noqa: PLC0415
    except ImportError as err:
        raise BuildError(
            f"TensorRT não encontrado ({err}); compile na Jetson ou use --cpu-only"
        ) from err
    return TrtCompiler()


def parse_args(argv: Sequence[str] | None) -> BuildArgs:
    parser = _parser()
    namespace = parser.parse_args(argv)
    raw_precisions: object = namespace.precision
    cpu_only: object = namespace.cpu_only
    force: object = namespace.force
    package: object = namespace.package
    engine_dir: object = namespace.engine_dir
    if not isinstance(cpu_only, bool) or not isinstance(force, bool):
        parser.error("flags inválidas")
    if not isinstance(package, Path) or not (engine_dir is None or isinstance(engine_dir, Path)):
        parser.error("caminhos inválidos")
    if raw_precisions is not None and cpu_only:
        parser.error("--precision não pode ser usado com --cpu-only")
    return BuildArgs(
        package=package,
        precisions=_precisions(raw_precisions),
        cpu_only=cpu_only,
        force=force,
        engine_dir=engine_dir,
    )


def _precisions(raw: object) -> tuple[Precision, ...]:
    if not isinstance(raw, list):
        return ALL_PRECISIONS
    chosen = {_to_precision(str(item)) for item in raw}
    return tuple(precision for precision in ALL_PRECISIONS if precision in chosen)


def _to_precision(value: str) -> Precision:
    match value:
        case "FP16":
            return "FP16"
        case "FP32":
            return "FP32"
        case _:
            raise ValueError(f"precisão inválida: {value}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Prepara a pasta do engine a partir de um pacote MODEL.",
    )
    parser.add_argument("package", type=Path, help="pasta do pacote com metadata.json")
    parser.add_argument(
        "--precision",
        nargs="+",
        choices=ALL_PRECISIONS,
        help="precisões a compilar (padrão: FP16 FP32)",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="só copia ONNX e metadados e grava o manifesto, sem compilar",
    )
    parser.add_argument("--force", action="store_true", help="substitui a pasta existente")
    parser.add_argument(
        "--engine-dir",
        type=Path,
        help="raiz dos engines (padrão: $P2_ENGINE_DIR ou /opt/ods/engines)",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
