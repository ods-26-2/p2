import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import cv2
import numpy as np

import p2
from yolo_demo import (
    DEFAULT_CONF,
    DEFAULT_IOU,
    EXIT_ERROR,
    EXIT_OK,
    Detection,
    Thresholds,
    class_names_for,
    detect,
    draw,
)

logger = logging.getLogger(__name__)

ASSETS: Final = Path(__file__).resolve().parent.parent / "assets"
DEFAULT_IMAGES: Final = (ASSETS / "bus.jpg", ASSETS / "dogi.jpeg")
DEFAULT_ENGINE_REF: Final = "app-plates-1.0.0"
DEFAULT_OUT_DIR: Final = Path("runs") / "example"


@dataclass(frozen=True, slots=True)
class ExampleArgs:
    engine_ref: str
    images: tuple[Path, ...]
    classes_file: Path | None
    out_dir: Path
    thresholds: Thresholds


def read_class_names(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run(args: ExampleArgs) -> int:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with p2.load(args.engine_ref) as handle:
        info = p2.info(handle)
        class_names = _class_names(args)
        logger.info("engine %s (%s)", info["engine_version"], info["precision"])
        logger.info("reconhece %d classes: %s", len(class_names), ", ".join(class_names))
        for image_path in args.images:
            loaded = cv2.imread(str(image_path))
            if loaded is None:
                logger.error("imagem ilegível: %s", image_path)
                return EXIT_ERROR
            image = loaded.astype(np.uint8)
            detections = detect(handle, image, class_names, args.thresholds)
            _report(image_path, detections)
            annotated = args.out_dir / f"{image_path.stem}-anotada.jpg"
            cv2.imwrite(str(annotated), draw(image, detections))
            logger.info("imagem anotada em %s", annotated)
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
    try:
        return run(parse_args(argv))
    except p2.P2Error as err:
        logger.error("%s", err)
        return EXIT_ERROR


def parse_args(argv: Sequence[str] | None) -> ExampleArgs:
    parser = argparse.ArgumentParser(description="Detecção YOLO com o p2 (padrão: app-plates).")
    parser.add_argument("images", type=Path, nargs="*", help="imagens (padrão: assets/)")
    parser.add_argument("--engine-ref", default=DEFAULT_ENGINE_REF)
    parser.add_argument("--classes", type=Path, help="um nome por linha (padrão: metadata.json)")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    namespace = parser.parse_args(argv)
    images: object = namespace.images
    engine_ref: object = namespace.engine_ref
    classes: object = namespace.classes
    conf: object = namespace.conf
    iou: object = namespace.iou
    out_dir: object = namespace.out_dir
    if not isinstance(images, list) or not all(isinstance(path, Path) for path in images):
        parser.error("imagens inválidas")
    if not isinstance(engine_ref, str) or not isinstance(out_dir, Path):
        parser.error("argumentos inválidos")
    if not (classes is None or isinstance(classes, Path)):
        parser.error("--classes inválido")
    if not isinstance(conf, float) or not isinstance(iou, float):
        parser.error("--conf e --iou devem ser números")
    return ExampleArgs(
        engine_ref=engine_ref,
        images=tuple(images) or DEFAULT_IMAGES,
        classes_file=classes,
        out_dir=out_dir,
        thresholds=Thresholds(conf=conf, iou=iou),
    )


def _class_names(args: ExampleArgs) -> list[str]:
    if args.classes_file is not None:
        return read_class_names(args.classes_file)
    return class_names_for(args.engine_ref)


def _report(image_path: Path, detections: Sequence[Detection]) -> None:
    logger.info("%s: %d detecções", image_path.name, len(detections))
    for detection in detections:
        x1, y1, x2, y2 = detection.box
        logger.info(
            "  %s %.2f [%.0f, %.0f, %.0f, %.0f]",
            detection.class_name,
            detection.score,
            x1,
            y1,
            x2,
            y2,
        )


if __name__ == "__main__":
    raise SystemExit(main())
