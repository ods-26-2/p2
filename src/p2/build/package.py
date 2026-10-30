import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import onnx
from google.protobuf.message import DecodeError
from onnx.helper import tensor_dtype_to_np_dtype
from typing_extensions import TypeIs

from p2.errors import BuildError
from p2.manifest import TensorSpec

METADATA_FILE: Final = "metadata.json"
SOURCE_PRECISION: Final = "FP32"
BOX_ROWS: Final = 4
CLASS_AXIS: Final = 1
HASH_CHUNK_BYTES: Final = 1 << 20


@dataclass(frozen=True, slots=True)
class ModelPackage:
    root: Path
    metadata_path: Path
    onnx_path: Path
    engine_ref: str
    sha256: str
    input_shape: tuple[int, ...]
    classes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OnnxIO:
    inputs: dict[str, TensorSpec]
    outputs: dict[str, TensorSpec]


def read_package(root: Path) -> ModelPackage:
    metadata_path = root / METADATA_FILE
    metadata = _load_metadata(metadata_path)
    precision = _require_str(metadata, "precision")
    if precision != SOURCE_PRECISION:
        raise BuildError(f"precisão do ONNX deve ser {SOURCE_PRECISION}, obtido {precision}")
    artifact_id = _require_str(metadata, "artifact_id")
    artifact_version = _require_str(metadata, "artifact_version")
    return ModelPackage(
        root=root,
        metadata_path=metadata_path,
        onnx_path=root / _require_str(metadata, "onnx_file"),
        engine_ref=f"{artifact_id}-{artifact_version}",
        sha256=_require_str(metadata, "hash_sha256").lower(),
        input_shape=_require_shape(metadata, "input_shape"),
        classes=_require_classes(metadata, "classes"),
    )


def validate_package(package: ModelPackage) -> OnnxIO:
    if not package.onnx_path.is_file():
        raise BuildError(f"arquivo ONNX não encontrado: {package.onnx_path}")
    actual = sha256_file(package.onnx_path)
    if actual != package.sha256:
        raise BuildError(f"hash diferente: esperado {package.sha256}, obtido {actual}")
    io = read_onnx_io(package.onnx_path)
    _check_input(package, io)
    _check_output(package, io)
    return io


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def read_onnx_io(path: Path) -> OnnxIO:
    try:
        model = onnx.load(str(path), load_external_data=False)
    except DecodeError as err:
        raise BuildError(f"ONNX inválido em {path}: {err}") from err
    initializers = {tensor.name for tensor in model.graph.initializer}
    inputs = [value for value in model.graph.input if value.name not in initializers]
    return OnnxIO(
        inputs={value.name: _tensor_spec(value) for value in inputs},
        outputs={value.name: _tensor_spec(value) for value in model.graph.output},
    )


def _tensor_spec(value: onnx.ValueInfoProto) -> TensorSpec:
    tensor_type = value.type.tensor_type
    shape: list[int] = []
    for axis, dim in enumerate(tensor_type.shape.dim):
        if not dim.HasField("dim_value"):
            raise BuildError(f"forma dinâmica em '{value.name}', eixo {axis}: '{dim.dim_param}'")
        shape.append(dim.dim_value)
    dtype = tensor_dtype_to_np_dtype(tensor_type.elem_type).name
    return {"shape": shape, "dtype": dtype}


def _check_input(package: ModelPackage, io: OnnxIO) -> None:
    if len(io.inputs) != 1:
        raise BuildError(f"ONNX deve ter 1 entrada, obtido {len(io.inputs)}")
    (spec,) = io.inputs.values()
    if tuple(spec["shape"]) != package.input_shape:
        raise BuildError(
            f"input_shape diferente: esperado {list(package.input_shape)}, obtido {spec['shape']}"
        )


def _check_output(package: ModelPackage, io: OnnxIO) -> None:
    if len(io.outputs) != 1:
        raise BuildError(f"ONNX deve ter 1 saída, obtido {len(io.outputs)}")
    (spec,) = io.outputs.values()
    expected = BOX_ROWS + len(package.classes)
    shape = spec["shape"]
    rows = shape[CLASS_AXIS] if len(shape) > CLASS_AXIS else None
    if rows != expected:
        raise BuildError(
            f"saída deve ter {expected} linhas ({BOX_ROWS} + {len(package.classes)} classes), "
            f"obtido forma {shape}"
        )


def _load_metadata(path: Path) -> Mapping[str, object]:
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as err:
        raise BuildError(f"{METADATA_FILE} não encontrado em {path.parent}") from err
    except json.JSONDecodeError as err:
        raise BuildError(f"{METADATA_FILE} inválido: {err}") from err
    if not isinstance(raw, dict):
        raise BuildError(f"{METADATA_FILE} deve conter um objeto JSON")
    return raw


def _require(metadata: Mapping[str, object], key: str) -> object:
    if key not in metadata:
        raise BuildError(f"campo '{key}' ausente em {METADATA_FILE}")
    return metadata[key]


def _require_str(metadata: Mapping[str, object], key: str) -> str:
    value = _require(metadata, key)
    if not isinstance(value, str) or not value:
        raise BuildError(f"campo '{key}' deve ser texto não vazio")
    return value


def _require_list(metadata: Mapping[str, object], key: str) -> list[object]:
    value = _require(metadata, key)
    if not isinstance(value, list) or not value:
        raise BuildError(f"campo '{key}' deve ser uma lista não vazia")
    return value


def _require_shape(metadata: Mapping[str, object], key: str) -> tuple[int, ...]:
    items = _require_list(metadata, key)
    dims = tuple(dim for dim in items if _is_positive_int(dim))
    if len(dims) != len(items):
        raise BuildError(f"campo '{key}' deve conter apenas inteiros positivos")
    return dims


def _require_classes(metadata: Mapping[str, object], key: str) -> tuple[str, ...]:
    items = _require_list(metadata, key)
    names = tuple(name for name in items if isinstance(name, str))
    if len(names) != len(items):
        raise BuildError(f"campo '{key}' deve conter apenas nomes (texto)")
    return names


def _is_positive_int(value: object) -> TypeIs[int]:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0
