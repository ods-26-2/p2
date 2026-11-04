import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import cv2
import numpy as np

import p2
from yolo_demo import EXIT_ERROR, EXIT_OK, Detection, detect, draw

logger = logging.getLogger(__name__)

ASSETS: Final = Path(__file__).resolve().parent.parent / "assets"
IMAGES: Final = (ASSETS / "bus.jpg", ASSETS / "dogi.jpeg")
CLASS_NAMES_FILE: Final = ASSETS / "coco80.txt"
DEFAULT_ENGINE_REF: Final = "yolo11n-1.0.0"
DEFAULT_OUT_DIR: Final = Path("runs") / "example"


def read_class_names(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run(engine_ref: str, out_dir: Path) -> int:
    class_names = read_class_names(CLASS_NAMES_FILE)
    out_dir.mkdir(parents=True, exist_ok=True)
    with p2.load(engine_ref) as handle:
        info = p2.info(handle)
        logger.info("engine %s (%s)", info["engine_version"], info["precision"])
        for image_path in IMAGES:
            loaded = cv2.imread(str(image_path))
            if loaded is None:
                logger.error("imagem ilegível: %s", image_path)
                return EXIT_ERROR
            image = loaded.astype(np.uint8)
            detections = detect(handle, image, class_names)
            _report(image_path, detections)
            annotated = out_dir / f"{image_path.stem}-anotada.jpg"
            cv2.imwrite(str(annotated), draw(image, detections))
            logger.info("imagem anotada em %s", annotated)
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Inferência YOLO com o p2 nas imagens de assets/.")
    parser.add_argument("--engine-ref", default=DEFAULT_ENGINE_REF)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    engine_ref: object = args.engine_ref
    out_dir: object = args.out_dir
    if not isinstance(engine_ref, str) or not isinstance(out_dir, Path):
        parser.error("argumentos inválidos")
    try:
        return run(engine_ref, out_dir)
    except p2.P2Error as err:
        logger.error("%s", err)
        return EXIT_ERROR


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
