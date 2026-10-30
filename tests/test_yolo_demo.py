from pathlib import Path
from typing import Final

import numpy as np
import pytest

import yolo_demo
from p2.cli.build import EXIT_OK, main

FIXTURE_DIR: Final = Path(__file__).parent / "fixtures" / "yolo11n-1.0.0"
BUS_IMAGE: Final = Path(__file__).parent.parent / "assets" / "bus.jpg"


def test_letterbox_padding_and_scale() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    boxed = yolo_demo.letterbox(image, 64)
    assert boxed.tensor.shape == (1, 3, 64, 64)
    assert boxed.scale == pytest.approx(0.32)
    assert (boxed.pad_x, boxed.pad_y) == (0.0, 16.0)


def test_postprocess_maps_boxes_to_original_frame() -> None:
    boxed = yolo_demo.Letterboxed(
        tensor=np.zeros((1, 3, 64, 64), dtype=np.float32), scale=0.5, pad_x=0.0, pad_y=16.0
    )
    output = np.array([[[32.0], [32.0], [10.0], [10.0], [0.9], [0.1]]], dtype=np.float32)
    detections = yolo_demo.postprocess(output, boxed, (64, 128), ["pessoa", "capacete"])
    assert len(detections) == 1
    assert detections[0].class_name == "pessoa"
    assert detections[0].box == pytest.approx((54.0, 22.0, 74.0, 42.0))


def test_postprocess_unknown_class_name() -> None:
    boxed = yolo_demo.Letterboxed(
        tensor=np.zeros((1, 3, 8, 8), dtype=np.float32), scale=1.0, pad_x=0.0, pad_y=0.0
    )
    output = np.array([[[4.0], [4.0], [2.0], [2.0], [0.1], [0.8]]], dtype=np.float32)
    detections = yolo_demo.postprocess(output, boxed, (8, 8), ["só uma"])
    assert detections[0].class_name == "classe_1"


def test_thresholds_filter_and_suppress() -> None:
    boxed = yolo_demo.Letterboxed(
        tensor=np.zeros((1, 3, 64, 64), dtype=np.float32), scale=1.0, pad_x=0.0, pad_y=0.0
    )
    output = np.array(
        [
            [
                [10.0, 11.0, 40.0],
                [10.0, 10.0, 40.0],
                [8.0, 8.0, 8.0],
                [8.0, 8.0, 8.0],
                [0.9, 0.8, 0.3],
            ]
        ],
        dtype=np.float32,
    )
    default = yolo_demo.postprocess(output, boxed, (64, 64), ["a"])
    strict = yolo_demo.postprocess(output, boxed, (64, 64), ["a"], yolo_demo.Thresholds(conf=0.5))
    assert [round(d.score, 1) for d in default] == [0.9, 0.3]
    assert [round(d.score, 1) for d in strict] == [0.9]


@pytest.mark.skipif(
    not (FIXTURE_DIR / "model.onnx").is_file(), reason="model.onnx do yolo11n ausente"
)
def test_demo_on_bus_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("INFO")
    engines = tmp_path / "engines"
    assert main([str(FIXTURE_DIR), "--cpu-only", "--engine-dir", str(engines)]) == EXIT_OK
    monkeypatch.setenv("P2_ENGINE_DIR", str(engines))
    monkeypatch.delenv("P2_BACKEND", raising=False)
    out = tmp_path / "anotada.jpg"
    args = ["yolo11n-1.0.0", str(BUS_IMAGE), "--out", str(out)]
    assert yolo_demo.main(args) == yolo_demo.EXIT_OK
    assert out.is_file()
    assert "bus 0.9" in caplog.text
    assert "person 0.9" in caplog.text


def test_demo_missing_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("P2_ENGINE_DIR", str(tmp_path))
    monkeypatch.delenv("P2_BACKEND", raising=False)
    assert yolo_demo.main(["nada-1.0.0", str(BUS_IMAGE)]) == yolo_demo.EXIT_ERROR
    assert "ENGINE_NOT_FOUND" in caplog.text


def test_demo_unreadable_image(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    assert yolo_demo.main(["x-1.0.0", str(tmp_path / "nada.jpg")]) == yolo_demo.EXIT_ERROR
    assert "imagem ilegível" in caplog.text
