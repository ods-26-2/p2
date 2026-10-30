import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import cv2
import numpy as np
import pytest

import p2
from helpers.yolo import Comparison, Detections, FloatArray, compare, detect, letterbox
from p2.api import BACKEND_ENV
from p2.manifest import ENGINE_DIR_ENV, Precision, engine_dir, read_manifest

pytest.importorskip("tensorrt")
pytest.importorskip("onnxruntime")

pytestmark = pytest.mark.jetson

ENGINE_ENV: Final = "P2_EQUIVALENCE_ENGINE"
IMAGES_ENV: Final = "P2_EQUIVALENCE_IMAGES"
REPORT_ENV: Final = "P2_EQUIVALENCE_REPORT"
DEFAULT_ENGINE: Final = "yolo11n-1.0.0"
DEFAULT_IMAGES: Final = Path(__file__).parent.parent.parent / "assets"
IMAGE_SUFFIXES: Final = (".jpg", ".jpeg", ".png")
TOLERANCES: Final[dict[Precision, tuple[float, float]]] = {
    "FP32": (0.98, 0.01),
    "FP16": (0.95, 0.03),
}


@dataclass(frozen=True, slots=True)
class ImageResult:
    image: str
    reference_count: int
    comparisons: dict[Precision, Comparison]


@pytest.fixture
def engine_setup(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> tuple[str, list[Path]]:
    engine_ref = os.environ.get(ENGINE_ENV, "")
    if not engine_ref:
        root: Path = request.getfixturevalue("compiled_yolo_root")
        monkeypatch.setenv(ENGINE_DIR_ENV, str(root))
        engine_ref = DEFAULT_ENGINE
    images_dir = Path(os.environ.get(IMAGES_ENV, str(DEFAULT_IMAGES)))
    images = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        pytest.skip(f"nenhuma imagem em {images_dir}")
    return engine_ref, images


def _run(
    engine_ref: str,
    backend: str,
    precision: Precision,
    tensors: list[FloatArray],
    monkeypatch: pytest.MonkeyPatch,
) -> list[Detections]:
    monkeypatch.setenv(BACKEND_ENV, backend)
    with p2.load(engine_ref, {"precision": precision}) as handle:
        (output_name,) = p2.info(handle)["output_shape"]
        return [detect(p2.infer(handle, tensor)[0][output_name]) for tensor in tensors]


def test_cpu_trt_fp32_trt_fp16_are_equivalent(
    engine_setup: tuple[str, list[Path]], monkeypatch: pytest.MonkeyPatch
) -> None:
    engine_ref, images = engine_setup
    tensors: list[FloatArray] = []
    for path in images:
        image = cv2.imread(str(path))
        assert image is not None, f"imagem ilegível: {path}"
        tensors.append(letterbox(image.astype(np.uint8)).tensor)
    reference = _run(engine_ref, "ort", "FP32", tensors, monkeypatch)
    candidates = {
        precision: _run(engine_ref, "trt", precision, tensors, monkeypatch)
        for precision in TOLERANCES
    }
    results = [
        ImageResult(
            image=path.name,
            reference_count=len(reference[index]),
            comparisons={
                precision: compare(reference[index], detections[index])
                for precision, detections in candidates.items()
            },
        )
        for index, path in enumerate(images)
    ]
    report_path = os.environ.get(REPORT_ENV, "")
    if report_path:
        _write_report(Path(report_path), engine_ref, results)
    failures = [
        f"{result.image} {precision}: {comparison}"
        for result in results
        for precision, comparison in result.comparisons.items()
        if not comparison.within(
            min_iou=TOLERANCES[precision][0], max_conf_diff=TOLERANCES[precision][1]
        )
    ]
    assert failures == []


def _write_report(path: Path, engine_ref: str, results: list[ImageResult]) -> None:
    manifest = read_manifest(engine_dir(engine_ref, os.environ))
    platform = ", ".join(f"{key} {value}" for key, value in manifest["platform"].items())
    lines = [
        f"# Equivalência · {engine_ref}",
        "",
        f"- Data: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- Plataforma: {platform}",
        f"- Imagens: {len(results)}",
        "- Referência: CPU (ONNX Runtime) FP32; limiar 0,25, NMS 0,45",
        "- Detecções a ±0,03 do limiar ignoradas",
        "",
        "| Engine | IoU mín. exigida | Δconf máx. permitida | IoU mín. obtida "
        "| Δconf máx. obtida | Sem par | Resultado |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for precision, (min_iou, max_diff) in TOLERANCES.items():
        comparisons = [result.comparisons[precision] for result in results]
        worst_iou = min(comparison.min_iou for comparison in comparisons)
        worst_diff = max(comparison.max_conf_diff for comparison in comparisons)
        unmatched = sum(c.unmatched_reference + c.unmatched_candidate for c in comparisons)
        ok = all(c.within(min_iou=min_iou, max_conf_diff=max_diff) for c in comparisons)
        lines.append(
            f"| TRT {precision} | {min_iou:.2f} | {max_diff:.2f} | {worst_iou:.4f} | "
            f"{worst_diff:.4f} | {unmatched} | {'ok' if ok else 'FALHA'} |"
        )
    lines += [
        "",
        "## Por imagem",
        "",
        "| Imagem | Detecções CPU | FP32 IoU / Δconf | FP16 IoU / Δconf |",
        "| --- | --- | --- | --- |",
    ]
    for result in results:
        cells = [
            f"{result.comparisons[p].min_iou:.4f} / {result.comparisons[p].max_conf_diff:.4f}"
            for p in ("FP32", "FP16")
        ]
        lines.append(f"| {result.image} | {result.reference_count} | {cells[0]} | {cells[1]} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
