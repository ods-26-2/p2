import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, Literal, TypeAlias, TypedDict

from typing_extensions import NotRequired, TypeIs

from p2.errors import ErrorCode, P2BaseError, P2Error

Precision: TypeAlias = Literal["FP16", "FP32"]

SCHEMA_VERSION: Final = 2
STATIC_SCHEMA_VERSION: Final = 1
SUPPORTED_SCHEMAS: Final = frozenset({STATIC_SCHEMA_VERSION, SCHEMA_VERSION})
DYNAMIC_DIM: Final = -1
STATIC_DTYPE: Final = "float32"
TENSOR_DTYPES: Final = frozenset({"float32", "int32", "int64", "bool"})
PROFILE_KEYS: Final = ("min", "opt", "max")
MANIFEST_FILE: Final = "manifest.json"
ENGINE_DIR_ENV: Final = "P2_ENGINE_DIR"
DEFAULT_ENGINE_ROOT: Final = Path("/opt/ods/engines")


class ShapeProfile(TypedDict):
    min: list[int]
    opt: list[int]
    max: list[int]


class TensorSpec(TypedDict):
    shape: list[int]
    dtype: str
    profile: NotRequired[ShapeProfile]


class SourceInfo(TypedDict):
    onnx_file: str
    sha256: str
    metadata_file: str


class VariantInfo(TypedDict):
    file: str
    engine_version: str
    sha256: str


class PlatformInfo(TypedDict, total=False):
    jetpack: str
    l4t: str
    tensorrt: str
    cuda: str


class BuildInfo(TypedDict):
    tool: str
    built_at: str
    tf32: bool
    workspace_mb: int


class Manifest(TypedDict):
    schema_version: int
    engine_ref: str
    source: SourceInfo
    inputs: dict[str, TensorSpec]
    outputs: dict[str, TensorSpec]
    max_batch: int
    variants: dict[Precision, VariantInfo]
    platform: PlatformInfo
    build: BuildInfo


def resolve_engine_root(override: Path | None, env: Mapping[str, str]) -> Path:
    if override is not None:
        return override
    from_env = env.get(ENGINE_DIR_ENV)
    if from_env:
        return Path(from_env)
    return DEFAULT_ENGINE_ROOT


def write_manifest(manifest: Manifest, directory: Path) -> Path:
    path = directory / MANIFEST_FILE
    text = json.dumps(manifest, indent=2, ensure_ascii=False)
    path.write_text(f"{text}\n", encoding="utf-8")
    return path


class ManifestFormatError(P2BaseError):
    pass


def as_precision(value: object) -> Precision | None:
    match value:
        case "FP16":
            return "FP16"
        case "FP32":
            return "FP32"
        case _:
            return None


def engine_dir(engine_ref: str, env: Mapping[str, str]) -> Path:
    return resolve_engine_root(None, env) / engine_ref


def read_manifest(directory: Path) -> Manifest:
    path = directory / MANIFEST_FILE
    if not directory.is_dir():
        raise P2Error(ErrorCode.ENGINE_NOT_FOUND, f"pasta do engine não existe: {directory}")
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as err:
        raise P2Error(ErrorCode.ENGINE_NOT_FOUND, f"manifesto ausente: {path}") from err
    except (OSError, json.JSONDecodeError) as err:
        raise P2Error(ErrorCode.ENGINE_NOT_FOUND, f"manifesto ilegível em {path}: {err}") from err
    try:
        return parse_manifest(raw)
    except ManifestFormatError as err:
        raise P2Error(ErrorCode.ENGINE_NOT_FOUND, f"manifesto inválido em {path}: {err}") from err


def parse_manifest(raw: object) -> Manifest:
    data = _object(raw, "manifesto")
    schema_version = _int(data, "schema_version")
    if schema_version not in SUPPORTED_SCHEMAS:
        supported = ", ".join(str(version) for version in sorted(SUPPORTED_SCHEMAS))
        raise ManifestFormatError(
            f"schema_version {schema_version} não suportado (esperado {supported})"
        )
    dynamic = schema_version == SCHEMA_VERSION
    inputs = _tensors(_object(_field(data, "inputs"), "inputs"), "inputs", dynamic=dynamic)
    _check_input_profiles(inputs)
    return {
        "schema_version": schema_version,
        "engine_ref": _str(data, "engine_ref"),
        "source": _source(_object(_field(data, "source"), "source")),
        "inputs": inputs,
        "outputs": _tensors(
            _object(_field(data, "outputs"), "outputs"), "outputs", dynamic=dynamic
        ),
        "max_batch": _int(data, "max_batch"),
        "variants": _variants(_object(_field(data, "variants"), "variants")),
        "platform": _platform(_object(_field(data, "platform"), "platform")),
        "build": _build(_object(_field(data, "build"), "build")),
    }


def _source(data: Mapping[str, object]) -> SourceInfo:
    return {
        "onnx_file": _str(data, "onnx_file"),
        "sha256": _str(data, "sha256"),
        "metadata_file": _str(data, "metadata_file"),
    }


def is_dynamic(shape: Sequence[int]) -> bool:
    return DYNAMIC_DIM in shape


def manifest_schema(inputs: Mapping[str, TensorSpec], outputs: Mapping[str, TensorSpec]) -> int:
    specs = [*inputs.values(), *outputs.values()]
    static = all(
        spec["shape"] and not is_dynamic(spec["shape"]) and spec["dtype"] == STATIC_DTYPE
        for spec in specs
    )
    return STATIC_SCHEMA_VERSION if static else SCHEMA_VERSION


def max_batch(inputs: Mapping[str, TensorSpec]) -> int:
    firsts = [_largest_shape(spec)[0] for spec in inputs.values() if spec["shape"]]
    return max(firsts, default=1)


def profile_problem(shape: Sequence[int], profile: ShapeProfile) -> str | None:
    bounds = [profile["min"], profile["opt"], profile["max"]]
    if any(len(dims) != len(shape) for dims in bounds):
        return f"min/opt/max devem ter {len(shape)} eixos como {list(shape)}"
    for axis, (dim, low, mid, high) in enumerate(zip(shape, *bounds, strict=True)):
        if dim != DYNAMIC_DIM and not low == mid == high == dim:
            return f"eixo {axis} é fixo em {dim}; min/opt/max devem repetir esse valor"
        if not 0 < low <= mid <= high:
            return f"eixo {axis} exige 0 < min <= opt <= max, obtido {low}, {mid}, {high}"
    return None


def _largest_shape(spec: TensorSpec) -> list[int]:
    profile = spec.get("profile")
    return spec["shape"] if profile is None else profile["max"]


def _tensors(data: Mapping[str, object], where: str, *, dynamic: bool) -> dict[str, TensorSpec]:
    if not data:
        raise ManifestFormatError(f"'{where}' vazio")
    parse = _dynamic_tensor if dynamic else _static_tensor
    return {name: parse(_object(spec, f"{where}.{name}")) for name, spec in data.items()}


def _static_tensor(data: Mapping[str, object]) -> TensorSpec:
    shape = _dims(_field(data, "shape"), "shape", allow_dynamic=False)
    if not shape:
        raise ManifestFormatError("'shape' deve conter inteiros positivos: []")
    return {"shape": shape, "dtype": _str(data, "dtype")}


def _dynamic_tensor(data: Mapping[str, object]) -> TensorSpec:
    shape = _dims(_field(data, "shape"), "shape", allow_dynamic=True)
    dtype = _str(data, "dtype")
    if dtype not in TENSOR_DTYPES:
        raise ManifestFormatError(f"dtype '{dtype}' não suportado")
    spec: TensorSpec = {"shape": shape, "dtype": dtype}
    if "profile" in data:
        spec["profile"] = _profile(_object(data["profile"], "profile"))
    return spec


def _check_input_profiles(inputs: Mapping[str, TensorSpec]) -> None:
    for name, spec in inputs.items():
        profile = spec.get("profile")
        if profile is None:
            if is_dynamic(spec["shape"]):
                raise ManifestFormatError(f"entrada '{name}' com eixo dinâmico sem 'profile'")
            continue
        problem = profile_problem(spec["shape"], profile)
        if problem is not None:
            raise ManifestFormatError(f"'profile' de '{name}': {problem}")


def _profile(data: Mapping[str, object]) -> ShapeProfile:
    bounds = {key: _dims(_field(data, key), key, allow_dynamic=False) for key in PROFILE_KEYS}
    return {"min": bounds["min"], "opt": bounds["opt"], "max": bounds["max"]}


def _dims(value: object, key: str, *, allow_dynamic: bool) -> list[int]:
    if not isinstance(value, list):
        raise ManifestFormatError(f"'{key}' deve ser uma lista")
    dims = [dim for dim in value if _is_dim(dim, allow_dynamic=allow_dynamic)]
    if len(dims) != len(value):
        raise ManifestFormatError(f"'{key}' deve conter inteiros positivos: {value}")
    return dims


def _is_dim(value: object, *, allow_dynamic: bool) -> TypeIs[int]:
    if not isinstance(value, int) or isinstance(value, bool):
        return False
    return value > 0 or (allow_dynamic and value == DYNAMIC_DIM)


def _variants(data: Mapping[str, object]) -> dict[Precision, VariantInfo]:
    variants: dict[Precision, VariantInfo] = {}
    for key, value in data.items():
        precision = as_precision(key)
        if precision is None:
            raise ManifestFormatError(f"variante desconhecida: '{key}'")
        variant = _object(value, f"variants.{key}")
        variants[precision] = {
            "file": _str(variant, "file"),
            "engine_version": _str(variant, "engine_version"),
            "sha256": _str(variant, "sha256"),
        }
    return variants


def _platform(data: Mapping[str, object]) -> PlatformInfo:
    platform: PlatformInfo = {}
    if "jetpack" in data:
        platform["jetpack"] = _str(data, "jetpack")
    if "l4t" in data:
        platform["l4t"] = _str(data, "l4t")
    if "tensorrt" in data:
        platform["tensorrt"] = _str(data, "tensorrt")
    if "cuda" in data:
        platform["cuda"] = _str(data, "cuda")
    return platform


def _build(data: Mapping[str, object]) -> BuildInfo:
    tf32 = _field(data, "tf32")
    if not isinstance(tf32, bool):
        raise ManifestFormatError("'tf32' deve ser booleano")
    return {
        "tool": _str(data, "tool"),
        "built_at": _str(data, "built_at"),
        "tf32": tf32,
        "workspace_mb": _int(data, "workspace_mb"),
    }


def _object(value: object, where: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ManifestFormatError(f"'{where}' deve ser um objeto")
    return {str(key): item for key, item in value.items()}


def _field(data: Mapping[str, object], key: str) -> object:
    if key not in data:
        raise ManifestFormatError(f"campo '{key}' ausente")
    return data[key]


def _str(data: Mapping[str, object], key: str) -> str:
    value = _field(data, key)
    if not isinstance(value, str) or not value:
        raise ManifestFormatError(f"campo '{key}' deve ser texto não vazio")
    return value


def _int(data: Mapping[str, object], key: str) -> int:
    value = _field(data, key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ManifestFormatError(f"campo '{key}' deve ser inteiro")
    return value
