import json
from pathlib import Path

import pytest

from p2.errors import ErrorCode, P2Error
from p2.manifest import (
    DEFAULT_ENGINE_ROOT,
    ENGINE_DIR_ENV,
    MANIFEST_FILE,
    STATIC_SCHEMA_VERSION,
    Manifest,
    ManifestFormatError,
    as_precision,
    engine_dir,
    parse_manifest,
    read_manifest,
    resolve_engine_root,
    write_manifest,
)


def _cpu_only_manifest() -> Manifest:
    return {
        "schema_version": STATIC_SCHEMA_VERSION,
        "engine_ref": "yolo11n-1.0.0",
        "source": {
            "onnx_file": "model.onnx",
            "sha256": "f3f88a57b89f447e639b86279cdb0c636719f8bf7fefab8f3e27c44536e48de1",
            "metadata_file": "metadata.json",
        },
        "inputs": {"images": {"shape": [1, 3, 640, 640], "dtype": "float32"}},
        "outputs": {"output0": {"shape": [1, 84, 8400], "dtype": "float32"}},
        "max_batch": 1,
        "variants": {},
        "platform": {},
        "build": {
            "tool": "p2-build 0.1.0",
            "built_at": "2026-10-30T12:00:00+00:00",
            "tf32": False,
            "workspace_mb": 1024,
        },
    }


def test_override_wins_over_env(tmp_path: Path) -> None:
    env = {ENGINE_DIR_ENV: "/from/env"}
    assert resolve_engine_root(tmp_path, env) == tmp_path


def test_env_used_without_override() -> None:
    env = {ENGINE_DIR_ENV: "/from/env"}
    assert resolve_engine_root(None, env) == Path("/from/env")


@pytest.mark.parametrize("env", [{}, {ENGINE_DIR_ENV: ""}])
def test_default_root_without_override_or_env(env: dict[str, str]) -> None:
    assert resolve_engine_root(None, env) == DEFAULT_ENGINE_ROOT


def test_write_manifest_round_trips(tmp_path: Path) -> None:
    manifest = _cpu_only_manifest()
    path = write_manifest(manifest, tmp_path)
    assert path == tmp_path / MANIFEST_FILE
    assert json.loads(path.read_text(encoding="utf-8")) == manifest


def _trt_manifest() -> Manifest:
    manifest = _cpu_only_manifest()
    manifest["variants"] = {
        "FP16": {"file": "fp16.engine", "engine_version": "yolo11n-1.0.0-fp16", "sha256": "a" * 64},
        "FP32": {"file": "fp32.engine", "engine_version": "yolo11n-1.0.0-fp32", "sha256": "b" * 64},
    }
    manifest["platform"] = {
        "jetpack": "6.2.1",
        "l4t": "36.4.7",
        "tensorrt": "10.3.0",
        "cuda": "12.6",
    }
    return manifest


@pytest.mark.parametrize("manifest", [_cpu_only_manifest(), _trt_manifest()])
def test_read_manifest_round_trips(tmp_path: Path, manifest: Manifest) -> None:
    write_manifest(manifest, tmp_path)
    assert read_manifest(tmp_path) == manifest


def test_engine_dir_uses_root_and_ref() -> None:
    env = {ENGINE_DIR_ENV: "/raiz"}
    assert engine_dir("app-epi-1.0.0", env) == Path("/raiz/app-epi-1.0.0")


@pytest.mark.parametrize(
    ("value", "expected"), [("FP16", "FP16"), ("FP32", "FP32"), ("fp16", None), (16, None)]
)
def test_as_precision(value: object, expected: str | None) -> None:
    assert as_precision(value) == expected


def test_read_manifest_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(P2Error, match="pasta do engine não existe") as error:
        read_manifest(tmp_path / "nada")
    assert error.value.code == ErrorCode.ENGINE_NOT_FOUND


def _mutated(path: tuple[str, ...], value: object) -> dict[str, object]:
    data: dict[str, object] = json.loads(json.dumps(_trt_manifest()))
    target = data
    for key in path[:-1]:
        nested = target[key]
        assert isinstance(nested, dict)
        target = nested
    target[path[-1]] = value
    return data


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("schema_version",), 3, "schema_version 3"),
        (("engine_ref",), "", "engine_ref"),
        (("source",), [], "source"),
        (("inputs",), {}, "inputs"),
        (("inputs", "images", "shape"), [1, 3, 0, 640], "inteiros positivos"),
        (("inputs", "images", "shape"), "1x3", "lista"),
        (("max_batch",), True, "max_batch"),
        (("variants", "INT8"), {"file": "x", "engine_version": "y", "sha256": "z"}, "INT8"),
        (("platform", "tensorrt"), 10, "tensorrt"),
        (("build", "tf32"), "não", "tf32"),
    ],
)
def test_parse_manifest_rejects(path: tuple[str, ...], value: object, message: str) -> None:
    with pytest.raises(ManifestFormatError, match=message):
        parse_manifest(_mutated(path, value))


def test_parse_manifest_missing_field() -> None:
    data = _mutated(("build",), {})
    with pytest.raises(ManifestFormatError, match="campo 'tf32' ausente"):
        parse_manifest(data)
