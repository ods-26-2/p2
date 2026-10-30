import sys
from pathlib import Path

import pytest

from p2.build import platform
from p2.build.platform import (
    format_cuda_version,
    parse_jetpack,
    parse_l4t,
    read_platform,
    short_tensorrt_version,
)

TEGRA_RELEASE_SAMPLE = (
    "# R36 (release), REVISION: 4.7, GCID: 42132812, BOARD: generic, EABI: aarch64, "
    "DATE: Thu Sep 18 22:54:44 UTC 2025\n"
    "# KERNEL_VARIANT: oot\n"
)


def test_parse_l4t() -> None:
    assert parse_l4t(TEGRA_RELEASE_SAMPLE) == "36.4.7"


def test_parse_l4t_unknown_format() -> None:
    assert parse_l4t("qualquer coisa") is None


@pytest.mark.parametrize(
    ("text", "expected"), [("6.2.1+b38", "6.2.1"), ("6.2.1\n", "6.2.1"), ("", None)]
)
def test_parse_jetpack(text: str, expected: str | None) -> None:
    assert parse_jetpack(text) == expected


@pytest.mark.parametrize(
    ("version", "expected"), [("10.3.0.30", "10.3.0"), ("10.3.0", "10.3.0"), ("10.3", "10.3")]
)
def test_short_tensorrt_version(version: str, expected: str) -> None:
    assert short_tensorrt_version(version) == expected


@pytest.mark.parametrize(
    ("version", "expected"), [(12060, "12.6"), (12020, "12.2"), (11080, "11.8")]
)
def test_format_cuda_version(version: int, expected: str) -> None:
    assert format_cuda_version(version) == expected


def test_read_platform_omits_unreadable_fields(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(platform, "TEGRA_RELEASE", Path("/nao/existe"))
    monkeypatch.setattr(platform, "JETPACK_QUERY", ("comando-que-nao-existe-p2",))
    monkeypatch.setitem(sys.modules, "cuda.cudart", None)
    assert read_platform("10.3.0.30") == {"tensorrt": "10.3.0"}
    assert "campo omitido" in caplog.text
    assert "'cuda' omitido" in caplog.text


def test_read_platform_reads_l4t_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    release = tmp_path / "nv_tegra_release"
    release.write_text(TEGRA_RELEASE_SAMPLE, encoding="utf-8")
    monkeypatch.setattr(platform, "TEGRA_RELEASE", release)
    monkeypatch.setattr(platform, "JETPACK_QUERY", ("comando-que-nao-existe-p2",))
    monkeypatch.setitem(sys.modules, "cuda.cudart", None)
    assert read_platform("10.3.0") == {"l4t": "36.4.7", "tensorrt": "10.3.0"}
