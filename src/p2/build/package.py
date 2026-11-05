import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Final, TypeAlias

import onnx
from google.protobuf.message import DecodeError
from onnx.helper import tensor_dtype_to_np_dtype
from typing_extensions import TypeIs

from p2.errors import BuildError
from p2.manifest import (
    DYNAMIC_DIM,
    PROFILE_KEYS,
    TENSOR_DTYPES,
    ShapeProfile,
    TensorSpec,
    is_dynamic,
    profile_problem,
)

METADATA_FILE: Final = "metadata.json"
SOURCE_PRECISION: Final = "FP32"
BOX_ROWS: Final = 4
CLASS_AXIS: Final = 1
HASH_CHUNK_BYTES: Final = 1 << 20

Shape: TypeAlias = tuple[int, ...]
DeclaredShapes: TypeAlias = Shape | dict[str, Shape]


class Task(str, Enum):
    DETECT = "detect"
    CLASSIFY = "classify"
    SEGMENT = "segment"
    OCR = "ocr"
    REID = "reid"
    SPEECH_ENCODER = "speech-encoder"
    SPEECH_DECODER = "speech-decoder"
    VAD = "vad"
    OTHER = "other"


DEFAULT_TASK: Final = Task.DETECT


@dataclass(frozen=True, slots=True)
class ModelPackage:
    root: Path
    metadata_path: Path
    onnx_path: Path
    engine_ref: str
    sha256: str
    input_shape: DeclaredShapes
    classes: tuple[str, ...]
    task: Task = DEFAULT_TASK
    dynamic_shapes: dict[str, ShapeProfile] = field(default_factory=dict)


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
    task = _task(metadata)
    return ModelPackage(
        root=root,
        metadata_path=metadata_path,
        onnx_path=root / _require_str(metadata, "onnx_file"),
        engine_ref=f"{artifact_id}-{artifact_version}",
        sha256=_require_str(metadata, "hash_sha256").lower(),
        input_shape=_declared_shapes(metadata, "input_shape"),
        classes=_classes(metadata, task),
        task=task,
        dynamic_shapes=_dynamic_shapes(metadata, "dynamic_shapes"),
    )


def validate_package(package: ModelPackage) -> OnnxIO:
    if not package.onnx_path.is_file():
        raise BuildError(f"arquivo ONNX não encontrado: {package.onnx_path}")
    actual = sha256_file(package.onnx_path)
    if actual != package.sha256:
        raise BuildError(f"hash diferente: esperado {package.sha256}, obtido {actual}")
    io = read_onnx_io(package.onnx_path)
    _check_dtypes(io)
    if package.task is Task.DETECT:
        _check_single_io(io)
    _check_profiles(package, io)
    _check_input_shapes(package, io)
    if package.task is Task.DETECT:
        _check_detection_rows(package, io)
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
    _check_embedded_weights(model, path)
    initializers = {tensor.name for tensor in model.graph.initializer}
    inputs = [value for value in model.graph.input if value.name not in initializers]
    return OnnxIO(
        inputs={value.name: _tensor_spec(value) for value in inputs},
        outputs={value.name: _tensor_spec(value) for value in model.graph.output},
    )


def pin_fixed_axes(model: bytes, profiles: Mapping[str, ShapeProfile]) -> bytes:
    if not profiles:
        return model
    proto = onnx.load_from_string(model)
    for value in proto.graph.input:
        profile = profiles.get(value.name)
        if profile is None:
            continue
        dims = value.type.tensor_type.shape.dim
        for dim, low, high in zip(dims, profile["min"], profile["max"], strict=True):
            if low == high:
                dim.dim_value = low
    return proto.SerializeToString()


def _check_embedded_weights(model: onnx.ModelProto, path: Path) -> None:
    if any(tensor.data_location == onnx.TensorProto.EXTERNAL for tensor in model.graph.initializer):
        raise BuildError(f"ONNX com pesos externos não é suportado: {path}")


def _tensor_spec(value: onnx.ValueInfoProto) -> TensorSpec:
    if not value.type.HasField("tensor_type"):
        raise BuildError(f"'{value.name}' não é um tensor; P2 só aceita tensores")
    tensor_type = value.type.tensor_type
    if not tensor_type.HasField("shape"):
        raise BuildError(f"forma ausente em '{value.name}' no ONNX; rode a inferência de formas")
    shape = [
        dim.dim_value if dim.HasField("dim_value") and dim.dim_value > 0 else DYNAMIC_DIM
        for dim in tensor_type.shape.dim
    ]
    dtype = tensor_dtype_to_np_dtype(tensor_type.elem_type).name
    return {"shape": shape, "dtype": dtype}


def _check_dtypes(io: OnnxIO) -> None:
    supported = ", ".join(sorted(TENSOR_DTYPES))
    for name, spec in (*io.inputs.items(), *io.outputs.items()):
        if spec["dtype"] not in TENSOR_DTYPES:
            raise BuildError(f"tipo {spec['dtype']} de '{name}' não suportado (use {supported})")


def _check_single_io(io: OnnxIO) -> None:
    if len(io.inputs) != 1:
        raise BuildError(f"ONNX deve ter 1 entrada, obtido {len(io.inputs)}")
    if len(io.outputs) != 1:
        raise BuildError(f"ONNX deve ter 1 saída, obtido {len(io.outputs)}")


def _check_profiles(package: ModelPackage, io: OnnxIO) -> None:
    for name, profile in package.dynamic_shapes.items():
        if name not in io.inputs:
            raise BuildError(f"dynamic_shapes cita '{name}', que não é entrada do ONNX")
        shape = io.inputs[name]["shape"]
        if not is_dynamic(shape):
            raise BuildError(f"entrada '{name}' não tem eixo dinâmico no ONNX: {shape}")
        problem = profile_problem(shape, profile)
        if problem is not None:
            raise BuildError(f"dynamic_shapes de '{name}' inválido: {problem}")
    for name, spec in io.inputs.items():
        if is_dynamic(spec["shape"]) and name not in package.dynamic_shapes:
            raise BuildError(
                f"entrada '{name}' tem eixo dinâmico {spec['shape']}; "
                "declare min/opt/max em dynamic_shapes"
            )


def _check_input_shapes(package: ModelPackage, io: OnnxIO) -> None:
    declared = package.input_shape
    if isinstance(declared, dict):
        if set(declared) != set(io.inputs):
            raise BuildError(
                f"input_shape cita {sorted(declared)}, entradas do ONNX {sorted(io.inputs)}"
            )
        for name, shape in declared.items():
            _check_shape(f"input_shape de '{name}'", shape, io.inputs[name]["shape"])
        return
    if len(io.inputs) != 1:
        raise BuildError(
            f"input_shape em lista exige 1 entrada; o ONNX tem {sorted(io.inputs)}: "
            "use {nome: forma}"
        )
    (spec,) = io.inputs.values()
    _check_shape("input_shape", declared, spec["shape"])


def _check_shape(what: str, declared: Shape, actual: list[int]) -> None:
    if declared != tuple(actual):
        raise BuildError(f"{what} diferente: esperado {list(declared)}, obtido {actual}")


def _check_detection_rows(package: ModelPackage, io: OnnxIO) -> None:
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


def _task(metadata: Mapping[str, object]) -> Task:
    if "task" not in metadata:
        return DEFAULT_TASK
    value = metadata["task"]
    for task in Task:
        if value == task.value:
            return task
    allowed = ", ".join(task.value for task in Task)
    raise BuildError(f"campo 'task' inválido: {value!r} (use {allowed})")


def _classes(metadata: Mapping[str, object], task: Task) -> tuple[str, ...]:
    if task is not Task.DETECT and "classes" not in metadata:
        return ()
    items = _require_list(metadata, "classes")
    names = tuple(name for name in items if isinstance(name, str))
    if len(names) != len(items):
        raise BuildError("campo 'classes' deve conter apenas nomes (texto)")
    return names


def _declared_shapes(metadata: Mapping[str, object], key: str) -> DeclaredShapes:
    value = _require(metadata, key)
    if isinstance(value, dict):
        if not value:
            raise BuildError(f"campo '{key}' deve ter ao menos uma entrada")
        return {
            str(name): _dims(dims, f"{key}.{name}", allow_dynamic=True)
            for name, dims in value.items()
        }
    shape = _dims(value, key, allow_dynamic=True)
    if not shape:
        raise BuildError(f"campo '{key}' deve ser uma lista não vazia")
    return shape


def _dynamic_shapes(metadata: Mapping[str, object], key: str) -> dict[str, ShapeProfile]:
    if key not in metadata:
        return {}
    value = metadata[key]
    if not isinstance(value, dict) or not value:
        raise BuildError(f"campo '{key}' deve ser um objeto {{entrada: {{min, opt, max}}}}")
    return {str(name): _profile(bounds, f"{key}.{name}") for name, bounds in value.items()}


def _profile(value: object, where: str) -> ShapeProfile:
    if not isinstance(value, dict) or set(value) != set(PROFILE_KEYS):
        raise BuildError(f"campo '{where}' deve ter exatamente min, opt e max")
    bounds = {key: _dims(value[key], f"{where}.{key}", allow_dynamic=False) for key in PROFILE_KEYS}
    return {"min": list(bounds["min"]), "opt": list(bounds["opt"]), "max": list(bounds["max"])}


def _dims(value: object, where: str, *, allow_dynamic: bool) -> Shape:
    if not isinstance(value, list):
        raise BuildError(f"campo '{where}' deve ser uma lista de inteiros")
    dims = tuple(dim for dim in value if _is_dim(dim, allow_dynamic=allow_dynamic))
    if len(dims) != len(value):
        kind = "inteiros positivos ou -1" if allow_dynamic else "inteiros positivos"
        raise BuildError(f"campo '{where}' deve conter apenas {kind}")
    return dims


def _is_dim(value: object, *, allow_dynamic: bool) -> TypeIs[int]:
    if not isinstance(value, int) or isinstance(value, bool):
        return False
    return value > 0 or (allow_dynamic and value == DYNAMIC_DIM)
