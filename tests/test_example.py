from pathlib import Path
from typing import Final

import pytest

import example
import yolo_demo
from helpers.yolo11n import YOLO11N_MISSING, YOLO11N_REF, yolo11n_available, yolo11n_package
from p2.cli.build import EXIT_OK, main

COCO_NAMES: Final = example.ASSETS / "coco80.txt"


def test_read_class_names() -> None:
    names = example.read_class_names(COCO_NAMES)
    assert len(names) == 80  # noqa: PLR2004
    assert names[0] == "person"


def test_defaults_target_app_plates() -> None:
    args = example.parse_args([])
    assert args.engine_ref == "app-plates-1.0.0"
    assert args.images == example.DEFAULT_IMAGES
    assert args.classes_file is None


def test_images_and_thresholds_from_cli(tmp_path: Path) -> None:
    image = tmp_path / "placa.jpg"
    args = example.parse_args([str(image), "--conf", "0.5", "--classes", str(COCO_NAMES)])
    assert args.images == (image,)
    assert args.thresholds.conf == 0.5  # noqa: PLR2004
    assert args.classes_file == COCO_NAMES


@pytest.fixture
def yolo_engines(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    engines = tmp_path / "engines"
    package = yolo11n_package(tmp_path / "pkg")
    assert main([str(package), "--cpu-only", "--engine-dir", str(engines)]) == EXIT_OK
    monkeypatch.setenv("P2_ENGINE_DIR", str(engines))
    monkeypatch.delenv("P2_BACKEND", raising=False)
    return engines


@pytest.mark.skipif(not yolo11n_available(), reason=YOLO11N_MISSING)
@pytest.mark.usefixtures("yolo_engines")
def test_example_names_from_engine_metadata(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("INFO")
    out_dir = tmp_path / "out"
    assert example.main(["--engine-ref", YOLO11N_REF, "--out-dir", str(out_dir)]) == EXIT_OK
    assert (out_dir / "bus-anotada.jpg").is_file()
    assert (out_dir / "dogi-anotada.jpg").is_file()
    assert "reconhece 80 classes: person, bicycle" in caplog.text
    assert "person 0.9" in caplog.text


@pytest.mark.skipif(not yolo11n_available(), reason=YOLO11N_MISSING)
@pytest.mark.usefixtures("yolo_engines")
def test_example_classes_file_and_image(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO")
    names = tmp_path / "nomes.txt"
    names.write_text("pessoa\n", encoding="utf-8")
    image = example.ASSETS / "bus.jpg"
    args = [str(image), "--engine-ref", YOLO11N_REF, "--classes", str(names)]
    assert example.main([*args, "--out-dir", str(tmp_path / "out")]) == EXIT_OK
    assert "reconhece 1 classes: pessoa" in caplog.text
    assert "pessoa 0.9" in caplog.text
    assert "dogi.jpeg" not in caplog.text


def test_example_missing_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("P2_ENGINE_DIR", str(tmp_path))
    monkeypatch.delenv("P2_BACKEND", raising=False)
    assert example.main(["--out-dir", str(tmp_path / "out")]) == yolo_demo.EXIT_ERROR
    assert "ENGINE_NOT_FOUND" in caplog.text
