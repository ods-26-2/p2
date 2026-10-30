import json
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import pytest

from helpers.model_package import ENGINE_REF
from p2.cli import bench
from p2.cli.bench import (
    BenchConfig,
    Probes,
    default_report_path,
    memory_summary,
    parse_duration,
    parse_meminfo,
    parse_power_mode,
    run_bench,
    summarize_latency,
)
from p2.manifest import PlatformInfo

MEMINFO_SAMPLE = "MemTotal:        7834412 kB\nMemFree:  1000 kB\nMemAvailable:    2097152 kB\n"


@pytest.mark.parametrize(
    ("text", "seconds"),
    [("90", 90.0), ("45s", 45.0), ("30m", 1800.0), ("1h", 3600.0), ("0.5s", 0.5), (" 2M ", 120.0)],
)
def test_parse_duration(text: str, seconds: float) -> None:
    assert parse_duration(text) == seconds


@pytest.mark.parametrize("text", ["", "2x", "-1m", "0", "m"])
def test_parse_duration_invalid(text: str) -> None:
    with pytest.raises(ValueError, match="duração"):
        parse_duration(text)


def test_parse_meminfo() -> None:
    assert parse_meminfo(MEMINFO_SAMPLE) == 2048.0  # noqa: PLR2004
    assert parse_meminfo("MemTotal: 1 kB") is None


def test_parse_power_mode() -> None:
    assert parse_power_mode("NV Power Mode: 15W\n0\n") == "15W"
    assert parse_power_mode("nada") is None


def test_summarize_latency() -> None:
    summary = summarize_latency([float(value) for value in range(1, 101)])
    assert summary["p50"] == pytest.approx(50.5)
    assert summary["p95"] == pytest.approx(95.05)
    assert summary["max"] == 100.0  # noqa: PLR2004
    assert summary["mean"] == pytest.approx(50.5)


def test_memory_summary() -> None:
    summary = memory_summary(4000.0, 3800.0, 3600.0)
    assert summary["fixed_cost_mb"] == 200.0  # noqa: PLR2004
    assert summary["peak_mb"] == 400.0  # noqa: PLR2004
    assert summary["limit_mb"] == pytest.approx(520.0)


def test_memory_summary_without_probe() -> None:
    summary = memory_summary(None, None, None)
    assert summary["fixed_cost_mb"] is None
    assert summary["limit_mb"] is None


def _probes(samples: list[float]) -> Probes:
    iterator: Iterator[float] = iter(samples)
    last = samples[-1]

    def memory() -> float | None:
        return next(iterator, last)

    def platform() -> PlatformInfo:
        return {"l4t": "36.4.7"}

    return Probes(memory=memory, power_mode=lambda: "15W", platform=platform)


@pytest.mark.usefixtures("synth_engine")
def test_run_bench_on_cpu() -> None:
    config = BenchConfig(engine_ref=ENGINE_REF, precision=None, duration_s=0.2, warmup=2)
    report = run_bench(config, _probes([4000.0, 3900.0, 3850.0, 3880.0]))
    assert report["engine_version"] == f"{ENGINE_REF}-fp32"
    assert report["precision"] == "FP32"
    assert report["iterations"] > 0
    assert report["fps_mean"] > 0.0
    assert report["warmup"] == 2  # noqa: PLR2004
    assert report["latency_ms"]["p50"] <= report["latency_ms"]["max"]
    assert report["memory"]["fixed_cost_mb"] == 100.0  # noqa: PLR2004
    assert report["memory"]["peak_mb"] == 150.0  # noqa: PLR2004
    assert report["power_mode"] == "15W"
    assert report["platform"] == {"l4t": "36.4.7"}


def test_default_report_path() -> None:
    now = datetime(2026, 10, 30, 14, 5, 9, tzinfo=timezone.utc)
    path = default_report_path("app-epi-1.0.0", "FP16", now)
    assert path == Path("reports/bench/app-epi-1.0.0-fp16-20261030-140509.json")


@pytest.mark.usefixtures("synth_engine")
def test_main_writes_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bench, "SYSTEM_PROBES", _probes([1000.0]))
    out = tmp_path / "bench.json"
    args = [ENGINE_REF, "--duration", "0.2s", "--warmup", "1", "--out", str(out)]
    assert bench.main(args) == bench.EXIT_OK
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["engine_ref"] == ENGINE_REF
    assert report["tool"].startswith("p2-bench ")


@pytest.mark.usefixtures("engine_root")
def test_main_missing_engine(caplog: pytest.LogCaptureFixture) -> None:
    assert bench.main(["nada-1.0.0", "--duration", "1s"]) == bench.EXIT_ERROR
    assert "ENGINE_NOT_FOUND" in caplog.text


@pytest.mark.parametrize("args", [["x", "--duration", "2x"], ["x", "--warmup", "-1"]])
def test_main_usage_errors(args: list[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        bench.main(args)
    assert raised.value.code == 2  # noqa: PLR2004


@pytest.mark.usefixtures("synth_engine")
def test_main_unwritable_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(bench, "SYSTEM_PROBES", _probes([1000.0]))
    blocker = tmp_path / "arquivo"
    blocker.write_text("", encoding="utf-8")
    args = [ENGINE_REF, "--duration", "0.1s", "--warmup", "0", "--out", str(blocker / "r.json")]
    assert bench.main(args) == bench.EXIT_ERROR
    assert "não foi possível gravar" in caplog.text
