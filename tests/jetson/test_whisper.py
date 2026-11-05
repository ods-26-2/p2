from pathlib import Path

import pytest

import p2
import whisper_demo
from helpers.whisper import (
    DECODER_REF,
    ENCODER_REF,
    EXPECTED_TEXT,
    SAMPLE,
    TOKENIZER,
    WHISPER_MISSING,
    decoder_package,
    encoder_package,
    whisper_available,
)
from p2.api import BACKEND_ENV
from p2.cli.build import EXIT_OK, main
from p2.manifest import ENGINE_DIR_ENV

pytest.importorskip("tensorrt")

pytestmark = [
    pytest.mark.jetson,
    pytest.mark.skipif(not whisper_available(), reason=WHISPER_MISSING),
]


def test_whisper_small_fp16_transcribes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engines = tmp_path / "engines"
    for package in (encoder_package(tmp_path / "enc"), decoder_package(tmp_path / "dec")):
        args = [str(package), "--precision", "FP16", "--engine-dir", str(engines)]
        assert main(args) == EXIT_OK
    monkeypatch.setenv(ENGINE_DIR_ENV, str(engines))
    monkeypatch.setenv(BACKEND_ENV, "trt")
    tokenizer = whisper_demo.read_tokenizer(TOKENIZER)
    samples = whisper_demo.read_wav(SAMPLE)
    with p2.load(ENCODER_REF) as encoder, p2.load(DECODER_REF) as decoder:
        assert p2.info(decoder)["precision"] == "FP16"
        text = whisper_demo.transcribe(
            encoder, decoder, samples, tokenizer=tokenizer, language="en"
        )
    assert EXPECTED_TEXT in text.lower()
