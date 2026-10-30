import logging
import re
import subprocess
from pathlib import Path
from typing import Final

from p2.manifest import PlatformInfo

logger = logging.getLogger(__name__)

TEGRA_RELEASE: Final = Path("/etc/nv_tegra_release")
JETPACK_QUERY: Final = ("dpkg-query", "-W", "-f=${Version}", "nvidia-jetpack")
L4T_PATTERN: Final = re.compile(r"R(\d+)\s*\(release\),\s*REVISION:\s*(\d+(?:\.\d+)*)")
VERSION_PREFIX: Final = re.compile(r"^\d+(?:\.\d+)*")
TENSORRT_COMPONENTS: Final = 3
CUDA_MAJOR_FACTOR: Final = 1000
CUDA_MINOR_FACTOR: Final = 10


def parse_l4t(text: str) -> str | None:
    match = L4T_PATTERN.search(text)
    if match is None:
        return None
    return f"{match.group(1)}.{match.group(2)}"


def parse_jetpack(text: str) -> str | None:
    match = VERSION_PREFIX.match(text.strip())
    return None if match is None else match.group()


def short_tensorrt_version(version: str) -> str:
    match = VERSION_PREFIX.match(version.strip())
    if match is None:
        return version.strip()
    return ".".join(match.group().split(".")[:TENSORRT_COMPONENTS])


def format_cuda_version(version: int) -> str:
    major = version // CUDA_MAJOR_FACTOR
    minor = (version % CUDA_MAJOR_FACTOR) // CUDA_MINOR_FACTOR
    return f"{major}.{minor}"


def read_platform(tensorrt_version: str | None) -> PlatformInfo:
    platform: PlatformInfo = {}
    jetpack = _read_jetpack()
    if jetpack is not None:
        platform["jetpack"] = jetpack
    l4t = _read_l4t()
    if l4t is not None:
        platform["l4t"] = l4t
    if tensorrt_version is not None:
        platform["tensorrt"] = short_tensorrt_version(tensorrt_version)
    cuda = _read_cuda()
    if cuda is not None:
        platform["cuda"] = cuda
    return platform


def _read_jetpack() -> str | None:
    try:
        result = subprocess.run(JETPACK_QUERY, capture_output=True, text=True, check=False)
    except OSError as err:
        logger.warning("versão do JetPack não lida (%s); campo omitido", err)
        return None
    version = parse_jetpack(result.stdout) if result.returncode == 0 else None
    if version is None:
        logger.warning("pacote nvidia-jetpack não encontrado; campo 'jetpack' omitido")
    return version


def _read_l4t() -> str | None:
    try:
        text = TEGRA_RELEASE.read_text(encoding="utf-8")
    except OSError as err:
        logger.warning("versão do L4T não lida (%s); campo omitido", err)
        return None
    version = parse_l4t(text)
    if version is None:
        logger.warning("formato inesperado em %s; campo 'l4t' omitido", TEGRA_RELEASE)
    return version


def _read_cuda() -> str | None:
    try:
        from cuda import cudart  # noqa: PLC0415
    except ImportError as err:
        logger.warning("cuda-python ausente (%s); campo 'cuda' omitido", err)
        return None
    error, version = cudart.cudaRuntimeGetVersion()
    if error != cudart.cudaError_t.cudaSuccess:
        logger.warning("cudaRuntimeGetVersion falhou (%s); campo 'cuda' omitido", error.name)
        return None
    return format_cuda_version(version)


def installed_tensorrt() -> str | None:
    try:
        import tensorrt  # noqa: PLC0415
    except ImportError:
        return None
    return tensorrt.__version__
