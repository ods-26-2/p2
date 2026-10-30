import numpy as np
import pytest

from helpers.yolo import (
    PAD_VALUE,
    Detections,
    FloatArray,
    compare,
    decode,
    detect,
    iou,
    letterbox,
    nms,
)


def _output(rows: list[tuple[float, float, float, float, list[float]]]) -> FloatArray:
    columns = [[*box, *scores] for *box, scores in rows]
    return np.array(columns, dtype=np.float32).T[np.newaxis]


def _detections(boxes: list[list[float]], scores: list[float], classes: list[int]) -> Detections:
    return Detections(
        np.array(boxes, dtype=np.float32),
        np.array(scores, dtype=np.float32),
        np.array(classes, dtype=np.int64),
    )


def test_letterbox_shape_padding_and_rgb() -> None:
    image = np.zeros((320, 640, 3), dtype=np.uint8)
    image[:, :, 0] = 255
    boxed = letterbox(image)
    assert boxed.tensor.shape == (1, 3, 640, 640)
    assert boxed.tensor.dtype == np.float32
    assert boxed.scale == 1.0
    assert boxed.pad == (0.0, 160.0)
    pad = np.float32(PAD_VALUE / 255.0)
    assert boxed.tensor[0, :, 0, 0] == pytest.approx([pad, pad, pad])
    assert boxed.tensor[0, :, 320, 320] == pytest.approx([0.0, 0.0, 1.0])


def test_decode_filters_and_converts_boxes() -> None:
    output = _output([(10, 20, 4, 8, [0.9, 0.1]), (50, 50, 10, 10, [0.1, 0.2])])
    detections = decode(output, conf_threshold=0.25)
    assert len(detections) == 1
    assert detections.boxes[0] == pytest.approx([8, 16, 12, 24])
    assert detections.classes.tolist() == [0]
    assert detections.scores[0] == pytest.approx(0.9)


def test_iou() -> None:
    box = np.array([0, 0, 10, 10], dtype=np.float32)
    boxes = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], dtype=np.float32)
    assert iou(box, boxes) == pytest.approx([1.0, 1 / 3, 0.0])


def test_nms_suppresses_same_class_only() -> None:
    detections = _detections(
        [[0, 0, 10, 10], [1, 0, 11, 10], [0, 0, 10, 10]], [0.9, 0.8, 0.7], [0, 0, 1]
    )
    kept = nms(detections, iou_threshold=0.45)
    assert kept.scores.tolist() == pytest.approx([0.9, 0.7])
    assert kept.classes.tolist() == [0, 1]


def test_detect_runs_decode_and_nms() -> None:
    output = _output([(5, 5, 10, 10, [0.9]), (5.5, 5, 10, 10, [0.85]), (40, 40, 10, 10, [0.6])])
    assert len(detect(output)) == 2  # noqa: PLR2004


def test_compare_identical() -> None:
    detections = _detections([[0, 0, 10, 10]], [0.9], [0])
    result = compare(detections, detections)
    assert result.matched == 1
    assert result.within(min_iou=0.98, max_conf_diff=0.01)


def test_compare_detects_shift_and_conf_diff() -> None:
    reference = _detections([[0, 0, 10, 10]], [0.9], [0])
    candidate = _detections([[1, 0, 11, 10]], [0.85], [0])
    result = compare(reference, candidate)
    assert result.min_iou == pytest.approx(9 / 11)
    assert result.max_conf_diff == pytest.approx(0.05)
    assert not result.within(min_iou=0.95, max_conf_diff=0.03)


def test_compare_ignores_detections_near_threshold() -> None:
    reference = _detections([[0, 0, 10, 10], [50, 50, 60, 60]], [0.9, 0.26], [0, 0])
    candidate = _detections([[0, 0, 10, 10]], [0.9], [0])
    result = compare(reference, candidate)
    assert result.unmatched_reference == 0
    assert result.within(min_iou=0.98, max_conf_diff=0.01)


def test_compare_counts_missing_and_extra() -> None:
    reference = _detections([[0, 0, 10, 10]], [0.9], [0])
    candidate = _detections([[0, 0, 10, 10], [30, 30, 40, 40]], [0.9, 0.8], [1, 0])
    result = compare(reference, candidate)
    assert result.unmatched_reference == 1
    assert result.unmatched_candidate == 2  # noqa: PLR2004
