import argparse
import json
import logging
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import cv2
import numpy as np
import numpy.typing as npt

import p2

logger = logging.getLogger(__name__)

FloatArray = npt.NDArray[np.float32]
ImageArray = npt.NDArray[np.uint8]

ENGINE_DIR_ENV: Final = "P2_ENGINE_DIR"
DEFAULT_ENGINE_ROOT: Final = "/opt/ods/engines"
PAD_VALUE: Final = 114
PIXEL_SCALE: Final = 255.0
BOX_ROWS: Final = 4
DEFAULT_CONF: Final = 0.25
DEFAULT_IOU: Final = 0.45
EXIT_OK: Final = 0
EXIT_ERROR: Final = 1
BOX_COLOR: Final = (0, 200, 0)


@dataclass(frozen=True, slots=True)
class Letterboxed:
    tensor: FloatArray
    scale: float
    pad_x: float
    pad_y: float


@dataclass(frozen=True, slots=True)
class Thresholds:
    conf: float = DEFAULT_CONF
    iou: float = DEFAULT_IOU


@dataclass(frozen=True, slots=True)
class Detection:
    class_id: int
    class_name: str
    score: float
    box: tuple[float, float, float, float]


def letterbox(image_bgr: ImageArray, size: int) -> Letterboxed:
    height, width = image_bgr.shape[:2]
    scale = min(size / height, size / width)
    new_width, new_height = round(width * scale), round(height * scale)
    resized = cv2.resize(image_bgr, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (size - new_width) / 2, (size - new_height) / 2
    top, left = round(pad_y - 0.1), round(pad_x - 0.1)
    canvas = np.full((size, size, 3), PAD_VALUE, dtype=np.uint8)
    canvas[top : top + new_height, left : left + new_width] = resized.astype(np.uint8)
    rgb = canvas[:, :, ::-1]
    chw = rgb.transpose(2, 0, 1)[np.newaxis]
    tensor = np.ascontiguousarray(chw, dtype=np.float32) / np.float32(PIXEL_SCALE)
    return Letterboxed(tensor=tensor, scale=scale, pad_x=pad_x, pad_y=pad_y)


def box_iou(box: FloatArray, boxes: FloatArray) -> FloatArray:
    top_left = np.maximum(box[:2], boxes[:, :2])
    bottom_right = np.minimum(box[2:], boxes[:, 2:])
    sizes = np.clip(bottom_right - top_left, 0, None)
    intersection = sizes[:, 0] * sizes[:, 1]
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = np.maximum(area + areas - intersection, np.float32(1e-9))
    return np.asarray(intersection / union, dtype=np.float32)


def nms(
    boxes: FloatArray, scores: FloatArray, classes: npt.NDArray[np.int64], iou: float
) -> list[int]:
    kept: list[int] = []
    for class_id in np.unique(classes):
        order = np.flatnonzero(classes == class_id)
        order = order[np.argsort(-scores[order], kind="stable")]
        while order.size:
            best, rest = int(order[0]), order[1:]
            kept.append(best)
            order = rest[box_iou(boxes[best], boxes[rest]) <= iou]
    return sorted(kept, key=lambda index: -float(scores[index]))


def postprocess(
    output: FloatArray,
    boxed: Letterboxed,
    image_size: tuple[int, int],
    class_names: Sequence[str],
    thresholds: Thresholds | None = None,
) -> list[Detection]:
    limits = thresholds or Thresholds()
    predictions = output[0].T
    class_scores = predictions[:, BOX_ROWS:]
    classes = class_scores.argmax(axis=1).astype(np.int64)
    scores = class_scores.max(axis=1).astype(np.float32)
    keep = scores >= limits.conf
    centers, scores, classes = predictions[keep, :BOX_ROWS], scores[keep], classes[keep]
    half = centers[:, 2:4] / np.float32(2)
    boxes = np.concatenate([centers[:, :2] - half, centers[:, :2] + half], axis=1)
    boxes = _to_original(boxes.astype(np.float32), boxed, image_size)
    return [
        Detection(
            class_id=int(classes[index]),
            class_name=_class_name(class_names, int(classes[index])),
            score=float(scores[index]),
            box=(
                float(boxes[index, 0]),
                float(boxes[index, 1]),
                float(boxes[index, 2]),
                float(boxes[index, 3]),
            ),
        )
        for index in nms(boxes, scores, classes, limits.iou)
    ]


def detect(
    handle: p2.Handle,
    image_bgr: ImageArray,
    class_names: Sequence[str],
    thresholds: Thresholds | None = None,
) -> list[Detection]:
    info = p2.info(handle)
    ((input_name, input_shape),) = info["input_shape"].items()
    ((output_name, _),) = info["output_shape"].items()
    boxed = letterbox(image_bgr, input_shape[2])
    outputs, stats = p2.infer(handle, {input_name: boxed.tensor})
    logger.info("infer em %.1f ms (%s)", stats["latency_ms"], info["engine_version"])
    height, width = image_bgr.shape[:2]
    output = np.asarray(outputs[output_name], dtype=np.float32)
    return postprocess(output, boxed, (height, width), class_names, thresholds)


def class_names_for(engine_ref: str) -> list[str]:
    root = Path(os.environ.get(ENGINE_DIR_ENV) or DEFAULT_ENGINE_ROOT)
    metadata: object = json.loads((root / engine_ref / "metadata.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        return []
    classes = metadata.get("classes")
    if not isinstance(classes, list):
        return []
    return [str(name) for name in classes]


def draw(image_bgr: ImageArray, detections: Sequence[Detection]) -> ImageArray:
    annotated = image_bgr.copy()
    for detection in detections:
        x1, y1, x2, y2 = (round(value) for value in detection.box)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), BOX_COLOR, 2)
        label = f"{detection.class_name} {detection.score:.2f}"
        cv2.putText(
            annotated, label, (x1, max(y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BOX_COLOR, 1
        )
    return annotated


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Detecção YOLO com o p2.")
    parser.add_argument("engine_ref", help="por exemplo app-epi-1.0.0")
    parser.add_argument("image", type=Path)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU)
    parser.add_argument("--out", type=Path, help="grava a imagem anotada")
    args = parser.parse_args(argv)
    engine_ref: object = args.engine_ref
    image_path: object = args.image
    conf: object = args.conf
    iou: object = args.iou
    out: object = args.out
    if not isinstance(engine_ref, str) or not isinstance(image_path, Path):
        parser.error("argumentos inválidos")
    if not isinstance(conf, float) or not isinstance(iou, float):
        parser.error("--conf e --iou devem ser números")
    loaded = cv2.imread(str(image_path))
    if loaded is None:
        logger.error("imagem ilegível: %s", image_path)
        return EXIT_ERROR
    image = loaded.astype(np.uint8)
    try:
        with p2.load(engine_ref) as handle:
            names = class_names_for(engine_ref)
            detections = detect(handle, image, names, Thresholds(conf=conf, iou=iou))
    except p2.P2Error as err:
        logger.error("%s", err)
        return EXIT_ERROR
    for detection in detections:
        x1, y1, x2, y2 = detection.box
        logger.info(
            "%s %.2f [%.0f, %.0f, %.0f, %.0f]",
            detection.class_name,
            detection.score,
            x1,
            y1,
            x2,
            y2,
        )
    if isinstance(out, Path):
        cv2.imwrite(str(out), draw(image, detections))
        logger.info("imagem anotada em %s", out)
    return EXIT_OK


def _to_original(boxes: FloatArray, boxed: Letterboxed, image_size: tuple[int, int]) -> FloatArray:
    height, width = image_size
    offset = np.array([boxed.pad_x, boxed.pad_y, boxed.pad_x, boxed.pad_y], dtype=np.float32)
    restored = (boxes - offset) / np.float32(boxed.scale)
    limits = np.array([width, height, width, height], dtype=np.float32)
    return np.clip(restored, 0, limits).astype(np.float32)


def _class_name(class_names: Sequence[str], class_id: int) -> str:
    return class_names[class_id] if class_id < len(class_names) else f"classe_{class_id}"


if __name__ == "__main__":
    raise SystemExit(main())
