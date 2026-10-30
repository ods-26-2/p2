import json
from pathlib import Path

import pytest

from helpers.model_package import ENGINE_REF
from p2.cli import list as list_cli
from p2.cli.list import compatibility, list_engines
from p2.manifest import MANIFEST_FILE, Manifest, read_manifest


def _with_variants(manifest: Manifest, tensorrt: str) -> Manifest:
    manifest["variants"] = {
        "FP16": {"file": "fp16.engine", "engine_version": "x-fp16", "sha256": "a" * 64},
        "FP32": {"file": "fp32.engine", "engine_version": "x-fp32", "sha256": "b" * 64},
    }
    manifest["platform"] = {"tensorrt": tensorrt}
    return manifest


def test_compatibility_cases(synth_engine: Path) -> None:
    cpu_only = read_manifest(synth_engine)
    assert compatibility(cpu_only, "10.3.0.30") == "só CPU (--cpu-only)"
    compiled = _with_variants(read_manifest(synth_engine), "10.3.0")
    assert compatibility(compiled, "10.3.0.30") == "compatível (TensorRT 10.3.0)"
    assert compatibility(compiled, None) == "sem TensorRT instalado (engine 10.3.0)"
    assert compatibility(compiled, "10.4.0").startswith(
        "incompatível: engine 10.3.0, instalado 10.4.0"
    )


def test_list_engines(synth_engine: Path, engine_root: Path) -> None:
    other = engine_root / "outro-2.0.0"
    other.mkdir()
    manifest = _with_variants(read_manifest(synth_engine), "10.3.0")
    manifest["engine_ref"] = "outro-2.0.0"
    (other / MANIFEST_FILE).write_text(json.dumps(manifest), encoding="utf-8")
    (engine_root / "quebrado").mkdir()
    (engine_root / ".synth-1.0.0.tmp-abc").mkdir()
    lines = list_engines(engine_root, "10.3.0.30")
    assert len(lines) == 3  # noqa: PLR2004
    assert (
        lines[0].startswith("outro-2.0.0") and "FP16,FP32" in lines[0] and "compatível" in lines[0]
    )
    assert lines[1].startswith("quebrado") and "manifesto inválido" in lines[1]
    assert lines[2].startswith(ENGINE_REF) and "só CPU" in lines[2]


@pytest.mark.usefixtures("synth_engine")
def test_main_prints_table(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO")
    assert list_cli.main([]) == list_cli.EXIT_OK
    assert "ENGINE_REF" in caplog.text
    assert ENGINE_REF in caplog.text


@pytest.mark.usefixtures("engine_root")
def test_main_empty_root(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO")
    assert list_cli.main([]) == list_cli.EXIT_OK
    assert "nenhum engine" in caplog.text


def test_main_missing_root(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    assert list_cli.main(["--engine-dir", str(tmp_path / "nada")]) == list_cli.EXIT_ERROR
    assert "não existe" in caplog.text
