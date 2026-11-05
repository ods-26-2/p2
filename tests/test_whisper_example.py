import json
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import pytest

import p2
import whisper_demo
import whisper_example
from helpers.whisper import (
    EXPECTED_TEXT,
    SAMPLE,
    TOKENIZER,
    WHISPER_MISSING,
    decoder_package,
    encoder_package,
    whisper_available,
)
from p2.cli.build import EXIT_OK, main

END_OF_TEXT: Final = 10
WORD: Final = 0
VOCABULARY: Final = 16


class FakeEngine:
    memory_mb: float = 100.0

    def __init__(self, engine_ref: str, options: Mapping[str, object] | None) -> None:
        self.engine_ref = engine_ref
        self.options = options
        self.calls = 0
        self.released = False


def _logits(token: int) -> npt.NDArray[np.float32]:
    logits = np.zeros((1, 1, VOCABULARY), dtype=np.float32)
    logits[0, 0, token] = 50.0
    return logits


@pytest.fixture
def fake_p2(monkeypatch: pytest.MonkeyPatch) -> list[FakeEngine]:
    engines: list[FakeEngine] = []

    def load(engine_ref: str, options: Mapping[str, object] | None = None) -> FakeEngine:
        engines.append(FakeEngine(engine_ref, options))
        return engines[-1]

    def infer(
        handle: FakeEngine, _: object
    ) -> tuple[dict[str, npt.NDArray[np.float32]], dict[str, float]]:
        handle.calls += 1
        if "encoder" in handle.engine_ref:
            return {"last_hidden_state": np.zeros(1, dtype=np.float32)}, {}
        token = WORD if handle.calls % 2 else END_OF_TEXT
        return {"logits": _logits(token)}, {}

    def release(handle: FakeEngine) -> None:
        handle.released = True

    monkeypatch.setattr("p2.load", load)
    monkeypatch.setattr("p2.infer", infer)
    monkeypatch.setattr("p2.release", release)
    return engines


@pytest.fixture
def bridge(tmp_path: Path) -> whisper_example.WhisperBridge:
    path = tmp_path / "tokenizer.json"
    names = ("<|endoftext|>", "<|startoftranscript|>", "<|pt|>", *whisper_demo.PROMPT_END)
    added = [{"id": END_OF_TEXT + index, "content": name} for index, name in enumerate(names)]
    data = {"model": {"vocab": {"oi": WORD}}, "added_tokens": added}
    path.write_text(json.dumps(data), encoding="utf-8")
    return whisper_example.WhisperBridge(tokenizer_path=path)


def _audio(seconds: float) -> npt.NDArray[np.float32]:
    return np.zeros(int(seconds * whisper_demo.SAMPLE_RATE), dtype=np.float32)


def test_load_ignores_i5_options_and_keeps_precision(
    bridge: whisper_example.WhisperBridge, fake_p2: list[FakeEngine]
) -> None:
    bridge.load("whisper-small-trt-fp16:1.0.0", {"device": "cuda", "torch_dtype": "float16"})
    assert [engine.options for engine in fake_p2] == [None, None]
    bridge.load("x", {"device": "cuda", "precision": "FP32"})
    assert fake_p2[-1].options == {"precision": "FP32"}
    assert [engine.engine_ref for engine in fake_p2[:2]] == [
        whisper_demo.DEFAULT_ENCODER,
        whisper_demo.DEFAULT_DECODER,
    ]


@pytest.mark.usefixtures("fake_p2")
def test_infer_returns_i5_format(bridge: whisper_example.WhisperBridge) -> None:
    handle = bridge.load()
    transcript, metrics = bridge.infer(handle, _audio(2.0))
    assert transcript["text"] == "oi"
    assert transcript["confidence"] == pytest.approx(1.0)
    assert transcript["token_probs"] == pytest.approx([1.0])
    assert set(metrics) == {"latency_ms", "gpu_memory_mb"}
    assert metrics["gpu_memory_mb"] == 200.0  # noqa: PLR2004


def test_long_audio_is_split_into_chunks(
    bridge: whisper_example.WhisperBridge, fake_p2: list[FakeEngine]
) -> None:
    handle = bridge.load()
    transcript, _ = bridge.infer(handle, _audio(61.0))
    assert fake_p2[0].calls == 3  # noqa: PLR2004
    assert transcript["text"] == "oi oi oi"


@pytest.mark.usefixtures("fake_p2")
@pytest.mark.parametrize("audio", [np.zeros(0, dtype=np.float32), [], {"audio": []}])
def test_empty_audio_is_invalid_input_shape(
    bridge: whisper_example.WhisperBridge, audio: object
) -> None:
    handle = bridge.load()
    with pytest.raises(p2.P2Error) as error:
        bridge.infer(handle, audio)
    assert error.value.code == p2.ErrorCode.INVALID_INPUT_SHAPE


@pytest.mark.usefixtures("fake_p2")
def test_accepts_dict_list_and_two_dimensions(bridge: whisper_example.WhisperBridge) -> None:
    handle = bridge.load()
    for audio in ({"audio": _audio(1.0)}, [0.0] * 100, _audio(1.0)[np.newaxis]):
        assert bridge.infer(handle, audio)[0]["text"] == "oi"


def test_release_frees_both_engines(
    bridge: whisper_example.WhisperBridge, fake_p2: list[FakeEngine]
) -> None:
    bridge.release(bridge.load())
    assert all(engine.released for engine in fake_p2)


def test_decoder_failure_releases_encoder(
    bridge: whisper_example.WhisperBridge,
    fake_p2: list[FakeEngine],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_load = p2.load

    def load(engine_ref: str, options: Mapping[str, object] | None = None) -> object:
        if "decoder" in engine_ref:
            raise p2.P2Error(p2.ErrorCode.OUT_OF_MEMORY, "simulado")
        return real_load(engine_ref, options)

    monkeypatch.setattr("p2.load", load)
    with pytest.raises(p2.P2Error):
        bridge.load()
    assert fake_p2[0].released


@pytest.mark.skipif(not whisper_available(), reason=WHISPER_MISSING)
def test_bridge_on_cpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engines = tmp_path / "engines"
    for package in (encoder_package(tmp_path / "enc"), decoder_package(tmp_path / "dec")):
        assert main([str(package), "--cpu-only", "--engine-dir", str(engines)]) == EXIT_OK
    monkeypatch.setenv("P2_ENGINE_DIR", str(engines))
    monkeypatch.setenv("P2_BACKEND", "ort")
    bridge = whisper_example.WhisperBridge(tokenizer_path=TOKENIZER, language="en")
    handle = bridge.load("whisper-small", {"device": "cuda"})
    transcript, _ = bridge.infer(handle, whisper_demo.read_wav(SAMPLE))
    bridge.release(handle)
    assert EXPECTED_TEXT in transcript["text"].lower()
    assert 0.0 < transcript["confidence"] <= 1.0
