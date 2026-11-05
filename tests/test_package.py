import json
from pathlib import Path

import pytest

from helpers.model_package import (
    CLASSES,
    ENGINE_REF,
    INPUT_NAME,
    INPUT_SHAPE,
    OUTPUT_NAME,
    PackageSpec,
    write_package,
)
from p2.build.package import read_package, sha256_file, validate_package
from p2.errors import BuildError

REQUIRED_FIELDS = (
    "artifact_id",
    "artifact_version",
    "onnx_file",
    "hash_sha256",
    "input_shape",
    "classes",
    "precision",
)


def test_valid_package_reads_metadata(tmp_path: Path) -> None:
    package = read_package(write_package(tmp_path))
    assert package.engine_ref == ENGINE_REF
    assert package.input_shape == INPUT_SHAPE
    assert package.classes == CLASSES
    assert package.onnx_path == tmp_path / "model.onnx"


def test_valid_package_returns_onnx_io(tmp_path: Path) -> None:
    io = validate_package(read_package(write_package(tmp_path)))
    assert io.inputs == {INPUT_NAME: {"shape": list(INPUT_SHAPE), "dtype": "float32"}}
    rows = 4 + len(CLASSES)
    assert io.outputs == {OUTPUT_NAME: {"shape": [1, rows, 4], "dtype": "float32"}}


def test_missing_metadata_file(tmp_path: Path) -> None:
    with pytest.raises(BuildError, match=r"metadata\.json não encontrado"):
        read_package(tmp_path)


@pytest.mark.parametrize("content", ["{nao json", "[1, 2]"])
def test_invalid_metadata_json(tmp_path: Path, content: str) -> None:
    (tmp_path / "metadata.json").write_text(content, encoding="utf-8")
    with pytest.raises(BuildError, match=r"metadata\.json"):
        read_package(tmp_path)


@pytest.mark.parametrize("key", REQUIRED_FIELDS)
def test_missing_field(tmp_path: Path, key: str) -> None:
    root = write_package(tmp_path, PackageSpec(metadata_drop=(key,)))
    with pytest.raises(BuildError, match=f"campo '{key}' ausente"):
        read_package(root)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("artifact_id", ""),
        ("artifact_version", 1),
        ("onnx_file", None),
        ("input_shape", []),
        ("input_shape", [1, 3, 0, 8]),
        ("input_shape", [1, True, 8, 8]),
        ("input_shape", "1,3,8,8"),
        ("classes", []),
        ("classes", {"0": "person"}),
        ("classes", ["person", 1]),
    ],
)
def test_wrong_field_type(tmp_path: Path, key: str, value: object) -> None:
    root = write_package(tmp_path, PackageSpec(metadata_overrides={key: value}))
    with pytest.raises(BuildError, match=f"campo '{key}'"):
        read_package(root)


def test_source_precision_must_be_fp32(tmp_path: Path) -> None:
    root = write_package(tmp_path, PackageSpec(metadata_overrides={"precision": "FP16"}))
    with pytest.raises(BuildError, match="deve ser FP32, obtido FP16"):
        read_package(root)


def test_missing_onnx_file(tmp_path: Path) -> None:
    root = write_package(tmp_path)
    (root / "model.onnx").unlink()
    with pytest.raises(BuildError, match="arquivo ONNX não encontrado"):
        validate_package(read_package(root))


def test_hash_mismatch(tmp_path: Path) -> None:
    root = write_package(tmp_path, PackageSpec(metadata_overrides={"hash_sha256": "0" * 64}))
    actual = sha256_file(root / "model.onnx")
    with pytest.raises(BuildError, match=f"hash diferente: esperado 0{{64}}, obtido {actual}"):
        validate_package(read_package(root))


def test_hash_is_case_insensitive(tmp_path: Path) -> None:
    root = write_package(tmp_path)
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["hash_sha256"] = metadata["hash_sha256"].upper()
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    validate_package(read_package(root))


def test_corrupt_onnx(tmp_path: Path) -> None:
    root = write_package(tmp_path)
    onnx_path = root / "model.onnx"
    onnx_path.write_bytes(b"garbage\x00\xff")
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["hash_sha256"] = sha256_file(onnx_path)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(BuildError, match="ONNX inválido"):
        validate_package(read_package(root))


def test_input_shape_mismatch(tmp_path: Path) -> None:
    root = write_package(tmp_path, PackageSpec(onnx_input_shape=(1, 3, 16, 16)))
    with pytest.raises(BuildError, match="input_shape diferente"):
        validate_package(read_package(root))


def test_dynamic_dimension_needs_profile(tmp_path: Path) -> None:
    root = write_package(tmp_path, PackageSpec(onnx_input_shape=("batch", 3, 8, 8)))
    with pytest.raises(BuildError, match="declare min/opt/max em dynamic_shapes"):
        validate_package(read_package(root))


def test_more_than_one_input(tmp_path: Path) -> None:
    root = write_package(tmp_path, PackageSpec(extra_input=True))
    with pytest.raises(BuildError, match="1 entrada"):
        validate_package(read_package(root))


def test_more_than_one_output(tmp_path: Path) -> None:
    root = write_package(tmp_path, PackageSpec(extra_output=True))
    with pytest.raises(BuildError, match="1 saída"):
        validate_package(read_package(root))


def test_output_rows_must_match_classes(tmp_path: Path) -> None:
    root = write_package(tmp_path, PackageSpec(output_rows=84))
    with pytest.raises(BuildError, match=r"7 linhas \(4 \+ 3 classes\)"):
        validate_package(read_package(root))
