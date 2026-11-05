from pathlib import Path
from typing import Final

import whisper_setup
from helpers.yolo11n import REPO_ROOT

MODELS: Final = REPO_ROOT / "models"
ENCODER_ONNX: Final = MODELS / whisper_setup.ENCODER_ONNX
DECODER_ONNX: Final = MODELS / whisper_setup.DECODER_ONNX
TOKENIZER: Final = MODELS / whisper_setup.TOKENIZER_FILE
SAMPLE: Final = MODELS / whisper_setup.SAMPLE_FILE
WHISPER_MISSING: Final = "models/whisper-* ausente; rode examples/whisper_setup.py"
EXPECTED_TEXT: Final = "ask not what your country can do for you"
ENCODER_REF: Final = "whisper-small-encoder-1.0.0"
DECODER_REF: Final = "whisper-small-decoder-1.0.0"


def whisper_available() -> bool:
    return all(path.is_file() for path in (ENCODER_ONNX, DECODER_ONNX, TOKENIZER, SAMPLE))


def encoder_package(directory: Path) -> Path:
    return whisper_setup.write_package(directory, whisper_setup.ENCODER, ENCODER_ONNX)


def decoder_package(directory: Path) -> Path:
    return whisper_setup.write_package(directory, whisper_setup.DECODER, DECODER_ONNX)
