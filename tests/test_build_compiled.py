import hashlib
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import pytest

from helpers.dynamic_package import DYNAMIC_SHAPES, write_dynamic_package
from helpers.model_package import ENGINE_REF, write_package
from p2.build.output import BuildStamp, write_compiled
from p2.build.package import OnnxIO, read_package, validate_package
from p2.errors import BuildError
from p2.manifest import (
    MANIFEST_FILE,
    Manifest,
    PlatformInfo,
    Precision,
    ShapeProfile,
    read_manifest,
)

PLATFORM: PlatformInfo = {"jetpack": "6.2.1", "l4t": "36.4.7", "tensorrt": "10.3.0", "cuda": "12.6"}
STAMP = BuildStamp(built_at=datetime(2026, 10, 30, tzinfo=timezone.utc), tool_version="0.1.0")


class FakeCompiler:
    def __init__(
        self,
        *,
        io: OnnxIO | None = None,
        fail_on: Precision | None = None,
        smoke_fails_on: Precision | None = None,
    ) -> None:
        self.calls: list[tuple[str, str, Precision]] = []
        self.smoke_calls: list[tuple[bool, Precision]] = []
        self.profiles: list[dict[str, ShapeProfile]] = []
        self._io = io
        self._fail_on = fail_on
        self._smoke_fails_on = smoke_fails_on

    def compile(
        self,
        onnx_path: Path,
        engine_path: Path,
        precision: Precision,
        profiles: Mapping[str, ShapeProfile],
    ) -> OnnxIO:
        self.calls.append((onnx_path.name, engine_path.name, precision))
        self.profiles.append(dict(profiles))
        if precision == self._fail_on:
            raise BuildError(f"compilação {precision} falhou: erro simulado do TensorRT")
        engine_path.write_bytes(f"engine-{precision}".encode())
        assert self._io is not None
        return self._io

    def platform(self) -> PlatformInfo:
        return PLATFORM.copy()

    def smoke_test(self, directory: Path, manifest: Manifest, precision: Precision) -> None:
        engine_written = (directory / manifest["variants"][precision]["file"]).is_file()
        self.smoke_calls.append((engine_written, precision))
        if precision == self._smoke_fails_on:
            raise BuildError(f"infer de teste {precision} com entrada zerada falhou: simulado")


def _package(tmp_path: Path) -> tuple[Path, OnnxIO]:
    root = write_package(tmp_path / "pkg")
    return root, validate_package(read_package(root))


def _write(
    tmp_path: Path,
    compiler: FakeCompiler,
    precisions: tuple[Precision, ...] = ("FP16", "FP32"),
    *,
    force: bool = False,
) -> Path:
    root, io = _package(tmp_path)
    return write_compiled(
        read_package(root),
        io,
        engine_root=tmp_path / "engines",
        force=force,
        stamp=STAMP,
        precisions=precisions,
        compiler=compiler,
    )


def test_compiles_each_precision_into_engine_dir(tmp_path: Path) -> None:
    _, io = _package(tmp_path)
    compiler = FakeCompiler(io=io)
    engine_dir = _write(tmp_path, compiler)
    assert compiler.calls == [
        ("model.onnx", "fp16.engine", "FP16"),
        ("model.onnx", "fp32.engine", "FP32"),
    ]
    assert sorted(path.name for path in engine_dir.iterdir()) == [
        "fp16.engine",
        "fp32.engine",
        MANIFEST_FILE,
        "metadata.json",
        "model.onnx",
    ]


def test_manifest_has_variants_and_platform(tmp_path: Path) -> None:
    _, io = _package(tmp_path)
    engine_dir = _write(tmp_path, FakeCompiler(io=io))
    manifest = read_manifest(engine_dir)
    assert manifest["variants"] == {
        "FP16": {
            "file": "fp16.engine",
            "engine_version": f"{ENGINE_REF}-fp16",
            "sha256": hashlib.sha256(b"engine-FP16").hexdigest(),
        },
        "FP32": {
            "file": "fp32.engine",
            "engine_version": f"{ENGINE_REF}-fp32",
            "sha256": hashlib.sha256(b"engine-FP32").hexdigest(),
        },
    }
    assert manifest["platform"] == PLATFORM
    assert manifest["build"]["tf32"] is False


def test_single_precision(tmp_path: Path) -> None:
    _, io = _package(tmp_path)
    engine_dir = _write(tmp_path, FakeCompiler(io=io), ("FP32",))
    assert set(read_manifest(engine_dir)["variants"]) == {"FP32"}
    assert not (engine_dir / "fp16.engine").exists()


def test_engine_io_mismatch_fails_and_writes_nothing(tmp_path: Path) -> None:
    _, io = _package(tmp_path)
    wrong = OnnxIO(inputs=io.inputs, outputs={"outra": {"shape": [1], "dtype": "float32"}})
    with pytest.raises(BuildError, match="diverge do ONNX"):
        _write(tmp_path, FakeCompiler(io=wrong))
    assert list((tmp_path / "engines").iterdir()) == []


def test_compile_failure_keeps_previous_dir(tmp_path: Path) -> None:
    _, io = _package(tmp_path)
    engine_dir = _write(tmp_path, FakeCompiler(io=io))
    marker = engine_dir / "marker"
    marker.write_text("antigo", encoding="utf-8")
    with pytest.raises(BuildError, match="erro simulado do TensorRT"):
        _write(tmp_path, FakeCompiler(io=io, fail_on="FP32"), force=True)
    assert marker.is_file()
    assert [path.name for path in (tmp_path / "engines").iterdir()] == [ENGINE_REF]


def test_manifest_is_valid_json_with_full_platform(tmp_path: Path) -> None:
    _, io = _package(tmp_path)
    engine_dir = _write(tmp_path, FakeCompiler(io=io))
    raw = json.loads((engine_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
    assert raw["platform"] == {
        "jetpack": "6.2.1",
        "l4t": "36.4.7",
        "tensorrt": "10.3.0",
        "cuda": "12.6",
    }


def test_smoke_test_runs_for_each_variant_after_manifest(tmp_path: Path) -> None:
    _, io = _package(tmp_path)
    compiler = FakeCompiler(io=io)
    _write(tmp_path, compiler)
    assert compiler.smoke_calls == [(True, "FP16"), (True, "FP32")]


def test_smoke_failure_publishes_nothing(tmp_path: Path) -> None:
    _, io = _package(tmp_path)
    with pytest.raises(BuildError, match="entrada zerada falhou"):
        _write(tmp_path, FakeCompiler(io=io, smoke_fails_on="FP16"))
    assert list((tmp_path / "engines").iterdir()) == []


def test_compiler_receives_dynamic_profiles(tmp_path: Path) -> None:
    package = read_package(write_dynamic_package(tmp_path / "pkg"))
    io = validate_package(package)
    compiler = FakeCompiler(io=io)
    target = write_compiled(
        package,
        io,
        engine_root=tmp_path / "engines",
        force=False,
        stamp=STAMP,
        precisions=("FP16",),
        compiler=compiler,
    )
    assert compiler.profiles == [DYNAMIC_SHAPES]
    assert read_manifest(target)["inputs"]["x"]["profile"] == DYNAMIC_SHAPES["x"]
