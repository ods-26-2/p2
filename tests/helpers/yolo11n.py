import shutil
from pathlib import Path
from typing import Final

REPO_ROOT: Final = Path(__file__).parent.parent.parent
YOLO11N_REF: Final = "yolo11n-1.0.0"
YOLO11N_ONNX: Final = REPO_ROOT / "models" / "yolo11n.onnx"
YOLO11N_METADATA: Final = REPO_ROOT / "tests" / "fixtures" / YOLO11N_REF / "metadata.json"
YOLO11N_MISSING: Final = "models/yolo11n.onnx ausente"


def yolo11n_available() -> bool:
    return YOLO11N_ONNX.is_file()


def yolo11n_package(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(YOLO11N_METADATA, directory / YOLO11N_METADATA.name)
    (directory / YOLO11N_ONNX.name).symlink_to(YOLO11N_ONNX)
    return directory
