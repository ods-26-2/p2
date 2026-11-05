import argparse
import json
from pathlib import Path

import pytest

import whisper_setup
from helpers.dynamic_package import write_dynamic_onnx
from p2.build.package import Task, read_onnx_io, read_package


def test_stages_use_valid_tasks() -> None:
    assert Task(whisper_setup.ENCODER.task) is Task.SPEECH_ENCODER
    assert Task(whisper_setup.DECODER.task) is Task.SPEECH_DECODER


def test_write_package_is_readable_by_p2_build(tmp_path: Path) -> None:
    onnx_path = tmp_path / whisper_setup.DECODER_LOGITS_ONNX
    write_dynamic_onnx(onnx_path)
    package = read_package(
        whisper_setup.write_package(tmp_path / "pkg", whisper_setup.DECODER, onnx_path)
    )
    assert package.engine_ref == "whisper-small-decoder-1.0.0"
    assert package.task is Task.SPEECH_DECODER
    assert package.onnx_path.is_file()
    assert package.sha256 == whisper_setup.sha256_of(onnx_path)
    assert package.dynamic_shapes["input_ids"]["max"] == [1, whisper_setup.MAX_TOKENS]


def test_write_package_twice_replaces_link(tmp_path: Path) -> None:
    onnx_path = tmp_path / whisper_setup.ENCODER_ONNX
    onnx_path.write_bytes(b"a")
    for _ in range(2):
        package = whisper_setup.write_package(tmp_path / "pkg", whisper_setup.ENCODER, onnx_path)
    metadata = json.loads((package / "metadata.json").read_text("utf-8"))
    assert metadata["onnx_file"] == whisper_setup.ENCODER_ONNX


def test_download_skips_file_with_matching_hash(tmp_path: Path) -> None:
    target = tmp_path / "a.bin"
    target.write_bytes(b"abc")
    remote = whisper_setup.RemoteFile("http://invalido", "a.bin", whisper_setup.sha256_of(target))
    assert whisper_setup.download(remote, tmp_path) == target


def test_download_rejects_wrong_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("subprocess.run", lambda *_, **__: None)
    (tmp_path / "a.bin").write_bytes(b"abc")
    remote = whisper_setup.RemoteFile("http://invalido", "a.bin", "0" * 64)
    with pytest.raises(ValueError, match="hash de"):
        whisper_setup.download(remote, tmp_path)


def test_keep_outputs_removes_the_others(tmp_path: Path) -> None:
    source, target = tmp_path / "a.onnx", tmp_path / "b.onnx"
    write_dynamic_onnx(source)
    assert whisper_setup.keep_outputs(source, target, frozenset({"y"})) == target
    assert set(read_onnx_io(target).outputs) == {"y"}
    assert set(read_onnx_io(source).outputs) == {"y", "mask", "next"}


def test_keep_outputs_reuses_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "b.onnx"
    target.write_bytes(b"pronto")
    assert whisper_setup.keep_outputs(tmp_path / "ausente.onnx", target, frozenset()) == target


def _args(tmp_path: Path, *, force: bool = False) -> argparse.Namespace:
    return argparse.Namespace(engine_dir=tmp_path, force=force, cpu_only=True, precision=["FP16"])


def test_compile_stage_skips_existing_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "whisper-small-encoder-1.0.0").mkdir()

    def fail(*_: object, **__: object) -> None:
        raise AssertionError("não devia compilar")

    monkeypatch.setattr("subprocess.run", fail)
    stage = whisper_setup.ENCODER
    assert whisper_setup.compile_stage(tmp_path, stage, _args(tmp_path)) == whisper_setup.EXIT_OK


def test_compile_stage_runs_in_separate_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[tuple[str, ...]] = []

    class Done:
        returncode = 0

    def run(command: tuple[str, ...], *, check: bool) -> Done:
        assert check is False
        commands.append(command)
        return Done()

    monkeypatch.setattr("subprocess.run", run)
    stage = whisper_setup.ENCODER
    (tmp_path / "whisper-small-encoder-1.0.0").mkdir()
    code = whisper_setup.compile_stage(tmp_path, stage, _args(tmp_path, force=True))
    assert code == whisper_setup.EXIT_OK
    assert commands[0][1:3] == ("-m", "p2.cli.build")
    assert "--force" in commands[0]
