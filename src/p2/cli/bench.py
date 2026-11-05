import argparse
import json
import logging
import os
import re
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Final, TypeAlias, TypedDict

import numpy as np

import p2
from p2.backends.base import TensorArray
from p2.build.platform import installed_tensorrt, read_platform
from p2.errors import P2Error
from p2.manifest import (
    STATIC_DTYPE,
    PlatformInfo,
    Precision,
    TensorSpec,
    as_precision,
    engine_dir,
    read_manifest,
)

logger = logging.getLogger(__name__)

PROG: Final = "p2-bench"
DISTRIBUTION: Final = "p2"
MEMINFO: Final = Path("/proc/meminfo")
NVPMODEL_QUERY: Final = ("nvpmodel", "-q")
DURATION_PATTERN: Final = re.compile(r"^(\d+(?:\.\d+)?)([smh]?)$")
MEMINFO_PATTERN: Final = re.compile(r"^MemAvailable:\s+(\d+)\s*kB", re.MULTILINE)
POWER_MODE_PATTERN: Final = re.compile(r"NV Power Mode:\s*(\S+)")
UNIT_SECONDS: Final[Mapping[str, float]] = {"": 1.0, "s": 1.0, "m": 60.0, "h": 3600.0}
KB_PER_MB: Final = 1024.0
LIMIT_FACTOR: Final = 1.3
MEMORY_SAMPLE_INTERVAL_S: Final = 0.5
RANDOM_SEED: Final = 0
DEFAULT_DURATION: Final = "30m"
DEFAULT_WARMUP: Final = 50
REPORT_DIR: Final = Path("reports") / "bench"
EXIT_OK: Final = 0
EXIT_ERROR: Final = 1

MemoryProbe: TypeAlias = Callable[[], float | None]
PowerProbe: TypeAlias = Callable[[], str | None]
PlatformProbe: TypeAlias = Callable[[], PlatformInfo]


class LatencySummary(TypedDict):
    p50: float
    p95: float
    max: float
    mean: float


class MemorySummary(TypedDict):
    available_before_mb: float | None
    available_after_load_mb: float | None
    available_min_mb: float | None
    fixed_cost_mb: float | None
    peak_mb: float | None
    limit_mb: float | None


class BenchReport(TypedDict):
    engine_ref: str
    engine_version: str
    precision: Precision
    duration_s: float
    warmup: int
    iterations: int
    fps_mean: float
    gpu_memory_mb: float
    latency_ms: LatencySummary
    memory: MemorySummary
    power_mode: str | None
    platform: PlatformInfo
    started_at: str
    tool: str


@dataclass(frozen=True, slots=True)
class BenchConfig:
    engine_ref: str
    precision: Precision | None
    duration_s: float
    warmup: int


@dataclass(frozen=True, slots=True)
class Probes:
    memory: MemoryProbe
    power_mode: PowerProbe
    platform: PlatformProbe


@dataclass(frozen=True, slots=True)
class Measurement:
    latencies: list[float]
    elapsed_s: float
    gpu_memory_mb: float
    available_min_mb: float | None


def parse_duration(text: str) -> float:
    match = DURATION_PATTERN.match(text.strip().lower())
    if match is None:
        raise ValueError(f"duração inválida: '{text}' (use 90, 45s, 30m ou 1h)")
    seconds = float(match.group(1)) * UNIT_SECONDS[match.group(2)]
    if seconds <= 0:
        raise ValueError(f"duração deve ser positiva: '{text}'")
    return seconds


def parse_meminfo(text: str) -> float | None:
    match = MEMINFO_PATTERN.search(text)
    return None if match is None else int(match.group(1)) / KB_PER_MB


def parse_power_mode(text: str) -> str | None:
    match = POWER_MODE_PATTERN.search(text)
    return None if match is None else match.group(1)


def summarize_latency(latencies: Sequence[float]) -> LatencySummary:
    values = np.asarray(latencies, dtype=np.float64)
    return {
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def memory_summary(
    before: float | None, after_load: float | None, minimum: float | None
) -> MemorySummary:
    fixed = None if before is None or after_load is None else before - after_load
    peak = None if before is None or minimum is None else before - minimum
    return {
        "available_before_mb": before,
        "available_after_load_mb": after_load,
        "available_min_mb": minimum,
        "fixed_cost_mb": fixed,
        "peak_mb": peak,
        "limit_mb": None if peak is None else peak * LIMIT_FACTOR,
    }


def run_bench(config: BenchConfig, probes: Probes) -> BenchReport:
    started_at = datetime.now(timezone.utc)
    options = None if config.precision is None else {"precision": config.precision}
    before = probes.memory()
    handle = p2.load(config.engine_ref, options)
    try:
        after_load = probes.memory()
        info = p2.info(handle)
        inputs = bench_inputs(read_manifest(engine_dir(config.engine_ref, os.environ))["inputs"])
        for _ in range(config.warmup):
            p2.infer(handle, inputs)
        measurement = _measure(handle, inputs, config.duration_s, probes.memory)
    finally:
        p2.release(handle)
    minimum = _lowest(after_load, measurement.available_min_mb)
    return {
        "engine_ref": config.engine_ref,
        "engine_version": info["engine_version"],
        "precision": info["precision"],
        "duration_s": measurement.elapsed_s,
        "warmup": config.warmup,
        "iterations": len(measurement.latencies),
        "fps_mean": len(measurement.latencies) / measurement.elapsed_s,
        "gpu_memory_mb": measurement.gpu_memory_mb,
        "latency_ms": summarize_latency(measurement.latencies),
        "memory": memory_summary(before, after_load, minimum),
        "power_mode": probes.power_mode(),
        "platform": probes.platform(),
        "started_at": started_at.isoformat(timespec="seconds"),
        "tool": f"{PROG} {version(DISTRIBUTION)}",
    }


def default_report_path(engine_ref: str, precision: Precision, now: datetime) -> Path:
    stamp = now.strftime("%Y%m%d-%H%M%S")
    return REPORT_DIR / f"{engine_ref}-{precision.lower()}-{stamp}.json"


def write_report(report: BenchReport, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{json.dumps(report, indent=2, ensure_ascii=False)}\n", encoding="utf-8")
    return path


def read_available_memory() -> float | None:
    try:
        return parse_meminfo(MEMINFO.read_text(encoding="utf-8"))
    except OSError:
        return None


def read_power_mode() -> str | None:
    try:
        result = subprocess.run(NVPMODEL_QUERY, capture_output=True, text=True, check=False)
    except OSError:
        return None
    return parse_power_mode(result.stdout) if result.returncode == 0 else None


def read_bench_platform() -> PlatformInfo:
    return read_platform(installed_tensorrt())


SYSTEM_PROBES: Final = Probes(
    memory=read_available_memory,
    power_mode=read_power_mode,
    platform=read_bench_platform,
)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config, out = parse_args(argv)
    logger.info(
        "medindo %s por %.0f s após %d de aquecimento",
        config.engine_ref,
        config.duration_s,
        config.warmup,
    )
    try:
        report = run_bench(config, SYSTEM_PROBES)
    except P2Error as err:
        logger.error("%s", err)
        return EXIT_ERROR
    now = datetime.now(timezone.utc)
    path = out or default_report_path(report["engine_ref"], report["precision"], now)
    try:
        write_report(report, path)
    except OSError as err:
        logger.error("não foi possível gravar o relatório em %s: %s", path, err)
        return EXIT_ERROR
    _log_summary(report, path)
    return EXIT_OK


def parse_args(argv: Sequence[str] | None) -> tuple[BenchConfig, Path | None]:
    parser = _parser()
    namespace = parser.parse_args(argv)
    engine_ref: object = namespace.engine_ref
    precision: object = namespace.precision
    duration: object = namespace.duration
    warmup: object = namespace.warmup
    out: object = namespace.out
    if not isinstance(engine_ref, str) or not isinstance(duration, float):
        parser.error("argumentos inválidos")
    if not isinstance(warmup, int) or warmup < 0:
        parser.error("--warmup deve ser um inteiro >= 0")
    if not (out is None or isinstance(out, Path)):
        parser.error("--out inválido")
    config = BenchConfig(
        engine_ref=engine_ref,
        precision=as_precision(precision),
        duration_s=duration,
        warmup=warmup,
    )
    return config, out


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROG, description="Mede latência e memória de um engine_ref."
    )
    parser.add_argument("engine_ref", help="por exemplo app-epi-1.0.0")
    parser.add_argument("--precision", choices=("FP16", "FP32"), help="padrão do backend")
    parser.add_argument(
        "--duration",
        type=_duration,
        default=parse_duration(DEFAULT_DURATION),
        help=f"tempo de medição: 90, 45s, 30m, 1h (padrão {DEFAULT_DURATION})",
    )
    parser.add_argument(
        "--warmup", type=int, default=DEFAULT_WARMUP, help="chamadas descartadas antes de medir"
    )
    parser.add_argument("--out", type=Path, help="arquivo JSON (padrão reports/bench/…)")
    return parser


def _duration(text: str) -> float:
    try:
        return parse_duration(text)
    except ValueError as err:
        raise argparse.ArgumentTypeError(str(err)) from err


def bench_inputs(specs: Mapping[str, TensorSpec]) -> dict[str, TensorArray]:
    rng = np.random.default_rng(RANDOM_SEED)
    return {name: _bench_input(spec, rng) for name, spec in specs.items()}


def _bench_input(spec: TensorSpec, rng: np.random.Generator) -> TensorArray:
    profile = spec.get("profile")
    shape = tuple(spec["shape"] if profile is None else profile["opt"])
    if spec["dtype"] == STATIC_DTYPE:
        return rng.random(shape, dtype=np.float32)
    return np.zeros(shape, dtype=np.dtype(spec["dtype"]))


def _measure(
    handle: p2.Handle,
    inputs: Mapping[str, TensorArray],
    duration_s: float,
    memory: MemoryProbe,
) -> Measurement:
    latencies: list[float] = []
    gpu_memory_mb = 0.0
    available_min: float | None = None
    start = time.perf_counter()
    deadline = start + duration_s
    next_sample = start
    now = start
    while now < deadline or not latencies:
        _, stats = p2.infer(handle, inputs)
        latencies.append(stats["latency_ms"])
        gpu_memory_mb = stats["gpu_memory_mb"]
        now = time.perf_counter()
        if now >= next_sample:
            available_min = _lowest(available_min, memory())
            next_sample = now + MEMORY_SAMPLE_INTERVAL_S
    elapsed = time.perf_counter() - start
    available_min = _lowest(available_min, memory())
    return Measurement(latencies, elapsed, gpu_memory_mb, available_min)


def _lowest(current: float | None, sample: float | None) -> float | None:
    if current is None:
        return sample
    if sample is None:
        return current
    return min(current, sample)


def _log_summary(report: BenchReport, path: Path) -> None:
    latency = report["latency_ms"]
    memory = report["memory"]
    logger.info(
        "%s: %d iterações, %.1f FPS, latência p50 %.2f ms, p95 %.2f ms, máx %.2f ms",
        report["engine_version"],
        report["iterations"],
        report["fps_mean"],
        latency["p50"],
        latency["p95"],
        latency["max"],
    )
    logger.info(
        "memória: handle %.1f MB, custo fixo %s MB, pico %s MB, limite %s MB",
        report["gpu_memory_mb"],
        _mb(memory["fixed_cost_mb"]),
        _mb(memory["peak_mb"]),
        _mb(memory["limit_mb"]),
    )
    logger.info("relatório gravado em %s", path)


def _mb(value: float | None) -> str:
    return "?" if value is None else f"{value:.1f}"


if __name__ == "__main__":
    raise SystemExit(main())
