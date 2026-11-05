import json
import wave
from pathlib import Path
from typing import Final

import numpy as np
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
from p2.cli.build import EXIT_OK, main

MEL_FRAMES: Final = 3000
TONE_HZ: Final = 1000.0


def _write_wav(path: Path, *, rate: int = 16000, channels: int = 1) -> None:
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(np.array([0, 16384, -16384], dtype=np.int16).tobytes())


def _tone(seconds: float = 1.0) -> whisper_demo.FloatArray:
    time = np.arange(int(whisper_demo.SAMPLE_RATE * seconds), dtype=np.float32)
    wave_values = np.sin(2 * np.pi * TONE_HZ * time / whisper_demo.SAMPLE_RATE)
    return np.asarray(wave_values, dtype=np.float32)


def test_read_wav_scales_to_unit_range(tmp_path: Path) -> None:
    path = tmp_path / "a.wav"
    _write_wav(path)
    np.testing.assert_allclose(whisper_demo.read_wav(path), [0.0, 0.5, -0.5])


@pytest.mark.parametrize(("rate", "channels"), [(44100, 1), (16000, 2)])
def test_read_wav_rejects_other_formats(tmp_path: Path, rate: int, channels: int) -> None:
    path = tmp_path / "a.wav"
    _write_wav(path, rate=rate, channels=channels)
    with pytest.raises(ValueError, match="mono, 16000 Hz"):
        whisper_demo.read_wav(path)


def test_log_mel_shape_and_range() -> None:
    mel = whisper_demo.log_mel(_tone())
    assert mel.shape == (1, whisper_demo.N_MELS, MEL_FRAMES)
    assert mel.dtype == np.float32
    assert float(mel.max()) <= 2.5  # noqa: PLR2004


def test_log_mel_truncates_after_thirty_seconds() -> None:
    assert whisper_demo.log_mel(_tone(31.0)).shape == (1, whisper_demo.N_MELS, MEL_FRAMES)


def test_tone_energy_lands_in_matching_mel_band() -> None:
    filters = whisper_demo.mel_filters()
    bin_index = int(TONE_HZ * whisper_demo.N_FFT / whisper_demo.SAMPLE_RATE)
    band = int(filters[:, bin_index].argmax())
    mel = whisper_demo.log_mel(_tone())
    assert int(mel[0, :, 10].argmax()) == band


def test_tokenizer_prompt_and_decode(tmp_path: Path) -> None:
    path = tmp_path / "tokenizer.json"
    vocab = {"ol": 0, "Ã¡": 1, "Ġmundo": 2}
    names = [
        "<|endoftext|>",
        "<|startoftranscript|>",
        "<|pt|>",
        "<|transcribe|>",
        "<|notimestamps|>",
    ]
    added = [{"id": 10 + index, "content": name} for index, name in enumerate(names)]
    path.write_text(
        json.dumps({"model": {"vocab": vocab}, "added_tokens": added}), encoding="utf-8"
    )
    tokenizer = whisper_demo.read_tokenizer(path)
    assert tokenizer.prompt("pt") == [11, 12, 13, 14]
    assert tokenizer.end_of_text == 10  # noqa: PLR2004
    assert tokenizer.decode([0, 1, 2, 10]) == "olá mundo"


@pytest.mark.skipif(not whisper_available(), reason=WHISPER_MISSING)
def test_transcribes_on_cpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engines = tmp_path / "engines"
    for package in (encoder_package(tmp_path / "enc"), decoder_package(tmp_path / "dec")):
        assert main([str(package), "--cpu-only", "--engine-dir", str(engines)]) == EXIT_OK
    monkeypatch.setenv("P2_ENGINE_DIR", str(engines))
    monkeypatch.setenv("P2_BACKEND", "ort")
    tokenizer = whisper_demo.read_tokenizer(TOKENIZER)
    samples = whisper_demo.read_wav(SAMPLE)
    with p2.load(ENCODER_REF) as encoder, p2.load(DECODER_REF) as decoder:
        text = whisper_demo.transcribe(
            encoder, decoder, samples, tokenizer=tokenizer, language="en"
        )
    assert EXPECTED_TEXT in text.lower()
