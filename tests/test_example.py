from pathlib import Path

import pytest

import example
import yolo_demo
from helpers.yolo11n import YOLO11N_MISSING, YOLO11N_REF, yolo11n_available, yolo11n_package
from p2.cli.build import EXIT_OK, main


def test_class_names_from_assets() -> None:
    names = example.read_class_names(example.CLASS_NAMES_FILE)
    assert len(names) == 80  # noqa: PLR2004
    assert names[0] == "person"


@pytest.mark.skipif(not yolo11n_available(), reason=YOLO11N_MISSING)
def test_example_on_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("INFO")
    engines = tmp_path / "engines"
    package = yolo11n_package(tmp_path / "pkg")
    assert main([str(package), "--cpu-only", "--engine-dir", str(engines)]) == EXIT_OK
    monkeypatch.setenv("P2_ENGINE_DIR", str(engines))
    monkeypatch.delenv("P2_BACKEND", raising=False)
    out_dir = tmp_path / "out"
    assert example.main(["--engine-ref", YOLO11N_REF, "--out-dir", str(out_dir)]) == EXIT_OK
    assert (out_dir / "bus-anotada.jpg").is_file()
    assert (out_dir / "dogi-anotada.jpg").is_file()
    assert "person 0.9" in caplog.text


def test_example_missing_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("P2_ENGINE_DIR", str(tmp_path))
    monkeypatch.delenv("P2_BACKEND", raising=False)
    assert example.main(["--out-dir", str(tmp_path / "out")]) == yolo_demo.EXIT_ERROR
    assert "ENGINE_NOT_FOUND" in caplog.text
