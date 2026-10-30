import argparse
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from p2.build.platform import installed_tensorrt, short_tensorrt_version
from p2.errors import P2Error
from p2.manifest import Manifest, read_manifest, resolve_engine_root

logger = logging.getLogger(__name__)

PROG: Final = "p2-list"
EXIT_OK: Final = 0
EXIT_ERROR: Final = 1
REF_WIDTH: Final = 28
VARIANTS_WIDTH: Final = 12


def compatibility(manifest: Manifest, installed: str | None) -> str:
    if not manifest["variants"]:
        return "só CPU (--cpu-only)"
    built = manifest["platform"].get("tensorrt", "?")
    if installed is None:
        return f"sem TensorRT instalado (engine {built})"
    current = short_tensorrt_version(installed)
    if current == built:
        return f"compatível (TensorRT {current})"
    return f"incompatível: engine {built}, instalado {current}; rode p2-build --force"


def describe(directory: Path, installed: str | None) -> str:
    try:
        manifest = read_manifest(directory)
    except P2Error as err:
        return _row(directory.name, "-", f"manifesto inválido: {err.detail}")
    variants = ",".join(sorted(manifest["variants"])) or "-"
    situation = compatibility(manifest, installed)
    return _row(manifest["engine_ref"], variants, situation)


def list_engines(root: Path, installed: str | None) -> list[str]:
    directories = sorted(
        path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")
    )
    return [describe(directory, installed) for directory in directories]


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(prog=PROG, description="Lista os engines instalados.")
    parser.add_argument(
        "--engine-dir",
        type=Path,
        help="raiz dos engines (padrão: $P2_ENGINE_DIR ou /opt/ods/engines)",
    )
    engine_dir: object = parser.parse_args(argv).engine_dir
    if not (engine_dir is None or isinstance(engine_dir, Path)):
        parser.error("--engine-dir inválido")
    root = resolve_engine_root(engine_dir, os.environ)
    if not root.is_dir():
        logger.error("raiz dos engines não existe: %s", root)
        return EXIT_ERROR
    lines = list_engines(root, installed_tensorrt())
    if not lines:
        logger.info("nenhum engine em %s", root)
        return EXIT_OK
    logger.info("%s", _row("ENGINE_REF", "VARIANTES", "SITUAÇÃO"))
    for line in lines:
        logger.info("%s", line)
    return EXIT_OK


def _row(engine_ref: str, variants: str, situation: str) -> str:
    return f"{engine_ref:<{REF_WIDTH}} {variants:<{VARIANTS_WIDTH}} {situation}"


if __name__ == "__main__":
    raise SystemExit(main())
