from dataclasses import dataclass
from typing import Final

import cv2
import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int64]
ImageArray = npt.NDArray[np.uint8]

INPUT_SIZE: Final = 640
PAD_VALUE: Final = 114
BOX_ROWS: Final = 4
CONF_THRESHOLD: Final = 0.25
NMS_IOU: Final = 0.45
MATCH_IOU: Final = 0.5
THRESHOLD_MARGIN: Final = 0.03
PIXEL_SCALE: Final = 255.0


@dataclass(frozen=True, slots=True)
class Letterboxed:
    tensor: FloatArray
    scale: float
    pad: tuple[float, float]


@dataclass(frozen=True, slots=True)
class Detections:
    boxes: FloatArray
    scores: FloatArray
    classes: IntArray

    def __len__(self) -> int:
        return int(self.scores.shape[0])

    def select(self, mask: npt.NDArray[np.bool_] | IntArray) -> "Detections":
        return Detections(self.boxes[mask], self.scores[mask], self.classes[mask])


@dataclass(frozen=True, slots=True)
class Comparison:
    matched: int
    min_iou: float
    max_conf_diff: float
    unmatched_reference: int
    unmatched_candidate: int

    def within(self, *, min_iou: float, max_conf_diff: float) -> bool:
        return (
            self.unmatched_reference == 0
            and self.unmatched_candidate == 0
            and self.min_iou >= min_iou
            and self.max_conf_diff <= max_conf_diff
        )


def letterbox(image_bgr: ImageArray, size: int = INPUT_SIZE) -> Letterboxed:
    height, width = image_bgr.shape[:2]
    scale = min(size / height, size / width)
    new_width, new_height = round(width * scale), round(height * scale)
    resized = cv2.resize(image_bgr, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (size - new_width) / 2, (size - new_height) / 2
    top, left = round(pad_y - 0.1), round(pad_x - 0.1)
    canvas = np.full((size, size, 3), PAD_VALUE, dtype=np.uint8)
    canvas[top : top + new_height, left : left + new_width] = resized.astype(np.uint8)
    rgb = canvas[:, :, ::-1].transpose(2, 0, 1)
    tensor = np.ascontiguousarray(rgb[np.newaxis], dtype=np.float32) / np.float32(PIXEL_SCALE)
    return Letterboxed(tensor=tensor, scale=scale, pad=(pad_x, pad_y))


def decode(output: FloatArray, *, conf_threshold: float = CONF_THRESHOLD) -> Detections:
    predictions = output[0].T
    class_scores = predictions[:, BOX_ROWS:]
    classes = class_scores.argmax(axis=1).astype(np.int64)
    scores = class_scores.max(axis=1).astype(np.float32)
    keep = scores >= conf_threshold
    centers = predictions[keep, :BOX_ROWS]
    half = centers[:, 2:4] / np.float32(2)
    boxes = np.concatenate([centers[:, :2] - half, centers[:, :2] + half], axis=1)
    return Detections(boxes.astype(np.float32), scores[keep], classes[keep])


def iou(box: FloatArray, boxes: FloatArray) -> FloatArray:
    top_left = np.maximum(box[:2], boxes[:, :2])
    bottom_right = np.minimum(box[2:], boxes[:, 2:])
    sizes = np.clip(bottom_right - top_left, 0, None)
    intersection = sizes[:, 0] * sizes[:, 1]
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area + areas - intersection
    return np.asarray(intersection / np.maximum(union, np.float32(1e-9)), dtype=np.float32)


def nms(detections: Detections, *, iou_threshold: float = NMS_IOU) -> Detections:
    kept: list[int] = []
    for class_id in np.unique(detections.classes):
        indices = np.flatnonzero(detections.classes == class_id)
        order = indices[np.argsort(-detections.scores[indices], kind="stable")]
        while order.size:
            best, rest = int(order[0]), order[1:]
            kept.append(best)
            overlaps = iou(detections.boxes[best], detections.boxes[rest])
            order = rest[overlaps <= iou_threshold]
    selected = np.array(sorted(kept, key=lambda index: -float(detections.scores[index])))
    return detections.select(selected.astype(np.int64))


def detect(output: FloatArray, *, conf_threshold: float = CONF_THRESHOLD) -> Detections:
    return nms(decode(output, conf_threshold=conf_threshold))


def compare(
    reference: Detections,
    candidate: Detections,
    *,
    conf_threshold: float = CONF_THRESHOLD,
) -> Comparison:
    strong = conf_threshold + THRESHOLD_MARGIN
    ious: list[float] = []
    conf_diffs: list[float] = []
    unmatched_reference = 0
    for index in np.flatnonzero(reference.scores >= strong):
        match = _best_match(reference, int(index), candidate)
        if match is None:
            unmatched_reference += 1
            continue
        overlap, other = match
        ious.append(overlap)
        conf_diffs.append(abs(float(reference.scores[index]) - float(candidate.scores[other])))
    unmatched_candidate = sum(
        1
        for index in np.flatnonzero(candidate.scores >= strong)
        if _best_match(candidate, int(index), reference) is None
    )
    return Comparison(
        matched=len(ious),
        min_iou=min(ious, default=1.0),
        max_conf_diff=max(conf_diffs, default=0.0),
        unmatched_reference=unmatched_reference,
        unmatched_candidate=unmatched_candidate,
    )


def _best_match(source: Detections, index: int, target: Detections) -> tuple[float, int] | None:
    same_class = np.flatnonzero(target.classes == source.classes[index])
    if same_class.size == 0:
        return None
    overlaps = iou(source.boxes[index], target.boxes[same_class])
    best = int(overlaps.argmax())
    if float(overlaps[best]) < MATCH_IOU:
        return None
    return float(overlaps[best]), int(same_class[best])
