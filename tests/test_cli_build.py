import json
import re
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Final

import pytest

from helpers.model_package import ENGINE_REF, PackageSpec, write_package
from p2.build.output import staged_engine_dir
from p2.build.package import read_package, sha256_file
from p2.cli.build import EXIT_BUILD_ERROR, EXIT_OK, main
from p2.errors import BuildError
from p2.manifest import ENGINE_DIR_ENV, MANIFEST_FILE

USAGE_EXIT: Final = 2
ISO_UTC: Final = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00$")


@pytest.fixture
def package_dir(tmp_path: Path) -> Path:
    return write_package(tmp_path / "pkg")


@pytest.fixture
def engine_root(tmp_path: Path) -> Path:
    return tmp_path / "engines"


def _run(*args: str | Path) -> int:
    return main([str(arg) for arg in args])


def _manifest(engine_dir: Path) -> dict[str, object]:
    loaded: object = json.loads((engine_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def test_cpu_only_writes_engine_dir(package_dir: Path, engine_root: Path) -> None:
    assert _run(package_dir, "--cpu-only", "--engine-dir", engine_root) == EXIT_OK
    engine_dir = engine_root / ENGINE_REF
    assert sorted(path.name for path in engine_dir.iterdir()) == [
        MANIFEST_FILE,
        "metadata.json",
        "model.onnx",
    ]
    assert (engine_dir / "model.onnx").read_bytes() == (package_dir / "model.onnx").read_bytes()
    assert [path.name for path in engine_root.iterdir()] == [ENGINE_REF]


def test_cpu_only_manifest_content(package_dir: Path, engine_root: Path) -> None:
    _run(package_dir, "--cpu-only", "--engine-dir", engine_root)
    manifest = _manifest(engine_root / ENGINE_REF)
    build = manifest.pop("build")
    assert manifest == {
        "schema_version": 1,
        "engine_ref": ENGINE_REF,
        "source": {
            "onnx_file": "model.onnx",
            "sha256": sha256_file(package_dir / "model.onnx"),
            "metadata_file": "metadata.json",
        },
        "inputs": {"images": {"shape": [1, 3, 8, 8], "dtype": "float32"}},
        "outputs": {"output0": {"shape": [1, 7, 4], "dtype": "float32"}},
        "max_batch": 1,
        "variants": {},
        "platform": {},
    }
    assert isinstance(build, dict)
    assert build["tool"] == f"p2-build {version('p2')}"
    assert ISO_UTC.match(build["built_at"])
    assert build["tf32"] is False
    assert build["workspace_mb"] == 1024  # noqa: PLR2004


def test_engine_dir_from_env(
    package_dir: Path, engine_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENGINE_DIR_ENV, str(engine_root))
    assert _run(package_dir, "--cpu-only") == EXIT_OK
    assert (engine_root / ENGINE_REF / MANIFEST_FILE).is_file()


def test_engine_dir_option_overrides_env(
    package_dir: Path, engine_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENGINE_DIR_ENV, str(tmp_path / "from-env"))
    assert _run(package_dir, "--cpu-only", "--engine-dir", engine_root) == EXIT_OK
    assert (engine_root / ENGINE_REF).is_dir()
    assert not (tmp_path / "from-env").exists()


def test_existing_dir_refused_without_force(
    package_dir: Path, engine_root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _run(package_dir, "--cpu-only", "--engine-dir", engine_root)
    marker = engine_root / ENGINE_REF / "marker"
    marker.write_text("antigo", encoding="utf-8")
    assert _run(package_dir, "--cpu-only", "--engine-dir", engine_root) == EXIT_BUILD_ERROR
    assert "pasta já existe" in caplog.text
    assert marker.is_file()


def test_force_replaces_existing_dir(package_dir: Path, engine_root: Path) -> None:
    _run(package_dir, "--cpu-only", "--engine-dir", engine_root)
    marker = engine_root / ENGINE_REF / "marker"
    marker.write_text("antigo", encoding="utf-8")
    assert _run(package_dir, "--cpu-only", "--force", "--engine-dir", engine_root) == EXIT_OK
    assert not marker.exists()
    assert (engine_root / ENGINE_REF / MANIFEST_FILE).is_file()
    assert [path.name for path in engine_root.iterdir()] == [ENGINE_REF]


def test_validation_failure_writes_nothing(tmp_path: Path, engine_root: Path) -> None:
    spec = PackageSpec(metadata_overrides={"hash_sha256": "0" * 64})
    package_dir = write_package(tmp_path / "pkg", spec)
    assert _run(package_dir, "--cpu-only", "--engine-dir", engine_root) == EXIT_BUILD_ERROR
    assert not engine_root.exists()


def test_failure_inside_staging_keeps_previous_dir(package_dir: Path, engine_root: Path) -> None:
    _run(package_dir, "--cpu-only", "--engine-dir", engine_root)
    marker = engine_root / ENGINE_REF / "marker"
    marker.write_text("antigo", encoding="utf-8")
    engine_ref = read_package(package_dir).engine_ref
    with (
        pytest.raises(BuildError, match="falha simulada"),
        staged_engine_dir(engine_root, engine_ref, force=True) as staging,
    ):
        (staging / "parcial").write_text("x", encoding="utf-8")
        raise BuildError("falha simulada")
    assert marker.is_file()
    assert [path.name for path in engine_root.iterdir()] == [ENGINE_REF]


def test_onnx_in_subfolder_is_flattened(tmp_path: Path, engine_root: Path) -> None:
    package_dir = write_package(tmp_path / "pkg", PackageSpec(onnx_name="sub/model.onnx"))
    assert _run(package_dir, "--cpu-only", "--engine-dir", engine_root) == EXIT_OK
    engine_dir = engine_root / ENGINE_REF
    assert (engine_dir / "model.onnx").is_file()
    source = _manifest(engine_dir)["source"]
    assert isinstance(source, dict)
    assert source["onnx_file"] == "model.onnx"


def test_precision_with_cpu_only_is_usage_error(package_dir: Path) -> None:
    with pytest.raises(SystemExit) as raised:
        _run(package_dir, "--cpu-only", "--precision", "FP16")
    assert raised.value.code == USAGE_EXIT


def test_invalid_precision_is_usage_error(package_dir: Path) -> None:
    with pytest.raises(SystemExit) as raised:
        _run(package_dir, "--precision", "INT8")
    assert raised.value.code == USAGE_EXIT


def test_without_tensorrt_is_build_error(
    package_dir: Path,
    engine_root: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "tensorrt", None)
    monkeypatch.delitem(sys.modules, "p2.build.compiler", raising=False)
    assert _run(package_dir, "--engine-dir", engine_root) == EXIT_BUILD_ERROR
    assert "TensorRT não encontrado" in caplog.text
    assert not engine_root.exists()


def test_unwritable_root_is_build_error(
    package_dir: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    blocker = tmp_path / "arquivo"
    blocker.write_text("", encoding="utf-8")
    assert _run(package_dir, "--cpu-only", "--engine-dir", blocker / "engines") == EXIT_BUILD_ERROR
    assert "não foi possível criar a pasta" in caplog.text
