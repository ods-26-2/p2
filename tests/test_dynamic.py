import json
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

import p2
from helpers.dynamic_package import (
    DYNAMIC_REF,
    DYNAMIC_SHAPES,
    FEATURES,
    INPUT_SHAPES,
    MAX_BATCH,
    MAX_TOKENS,
    ONNX_NAME,
    build_dynamic_engine,
    write_dynamic_onnx,
    write_dynamic_package,
    write_metadata,
)
from helpers.model_package import (
    ENGINE_REF,
    INPUT_NAME,
    INPUT_SHAPE,
    ONNX_IR_VERSION,
    ONNX_OPSET,
    PackageSpec,
    write_package,
)
from p2.build.output import io_matches
from p2.build.package import (
    OnnxIO,
    Task,
    pin_fixed_axes,
    read_onnx_io,
    read_package,
    validate_package,
)
from p2.cli.bench import bench_inputs
from p2.errors import BuildError, ErrorCode
from p2.manifest import (
    SCHEMA_VERSION,
    STATIC_SCHEMA_VERSION,
    ManifestFormatError,
    ShapeProfile,
    TensorSpec,
    manifest_schema,
    max_batch,
    parse_manifest,
    read_manifest,
)

SAMPLE_RATE: Final = 16000


def _feeds(batch: int = 4, tokens: int = 3) -> dict[str, npt.NDArray[np.generic]]:
    return {
        "x": np.ones((batch, FEATURES), dtype=np.float32),
        "tokens": np.arange(tokens, dtype=np.int64).reshape(1, tokens),
        "sr": np.array(SAMPLE_RATE, dtype=np.int64),
    }


@pytest.fixture
def dynamic_engine(engine_root: Path, tmp_path: Path) -> Path:
    return build_dynamic_engine(tmp_path / "dyn-pkg", engine_root)


def test_manifest_v2_has_profiles_and_dtypes(dynamic_engine: Path) -> None:
    manifest = read_manifest(dynamic_engine)
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["inputs"]["x"]["shape"] == [-1, FEATURES]
    assert manifest["inputs"]["x"]["dtype"] == "float32"
    assert manifest["inputs"]["x"].get("profile") == DYNAMIC_SHAPES["x"]
    assert manifest["inputs"]["tokens"]["dtype"] == "int64"
    assert manifest["inputs"]["sr"]["shape"] == []
    assert manifest["inputs"]["sr"]["dtype"] == "int64"
    assert manifest["outputs"]["mask"]["shape"] == [-1, FEATURES]
    assert manifest["outputs"]["mask"]["dtype"] == "bool"
    assert manifest["max_batch"] == MAX_BATCH


@pytest.mark.usefixtures("dynamic_engine")
def test_info_shows_dynamic_axes_and_keeps_five_fields() -> None:
    with p2.load(DYNAMIC_REF) as handle:
        assert p2.info(handle) == {
            "engine_version": f"{DYNAMIC_REF}-fp32",
            "input_shape": INPUT_SHAPES,
            "output_shape": {"y": [-1, FEATURES], "mask": [-1, FEATURES], "next": [1, -1]},
            "precision": "FP32",
            "max_batch": MAX_BATCH,
        }


@pytest.mark.usefixtures("dynamic_engine")
def test_batch_of_four_in_one_call() -> None:
    with p2.load(DYNAMIC_REF) as handle:
        outputs, _ = p2.infer(handle, _feeds(batch=4))
    assert outputs["y"].shape == (4, FEATURES)
    assert outputs["y"].dtype == np.float32
    np.testing.assert_array_equal(np.asarray(outputs["y"], dtype=np.float32), 2.0)


@pytest.mark.usefixtures("dynamic_engine")
def test_outputs_keep_tensor_dtype() -> None:
    with p2.load(DYNAMIC_REF) as handle:
        outputs, _ = p2.infer(handle, _feeds(tokens=3))
    assert outputs["mask"].dtype == np.bool_
    assert outputs["next"].dtype == np.int64
    expected = np.array([[SAMPLE_RATE, SAMPLE_RATE + 1, SAMPLE_RATE + 2]], dtype=np.int64)
    np.testing.assert_array_equal(np.asarray(outputs["next"], dtype=np.int64), expected)


@pytest.mark.usefixtures("dynamic_engine")
@pytest.mark.parametrize(("batch", "tokens"), [(1, 1), (MAX_BATCH, MAX_TOKENS), (3, 7)])
def test_output_shape_follows_each_call(batch: int, tokens: int) -> None:
    with p2.load(DYNAMIC_REF) as handle:
        outputs, _ = p2.infer(handle, _feeds(batch, tokens))
    assert outputs["y"].shape == (batch, FEATURES)
    assert outputs["next"].shape == (1, tokens)


@pytest.mark.usefixtures("dynamic_engine")
@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("x", np.ones((MAX_BATCH + 1, FEATURES), dtype=np.float32), "entre"),
        ("x", np.ones((0, FEATURES), dtype=np.float32), "entre"),
        ("x", np.ones((2, FEATURES + 1), dtype=np.float32), "entre"),
        ("x", np.ones((FEATURES,), dtype=np.float32), "entre"),
        ("tokens", np.zeros((1, MAX_TOKENS + 1), dtype=np.int64), "entre"),
        ("tokens", np.zeros((1, 2), dtype=np.int32), "int64"),
        ("tokens", np.zeros((1, 2), dtype=np.float32), "int64"),
        ("x", np.ones((2, FEATURES), dtype=np.float64), "float32"),
        ("sr", np.array([SAMPLE_RATE], dtype=np.int64), "forma"),
    ],
)
def test_outside_declared_is_invalid_input_shape(
    name: str, value: npt.NDArray[np.generic], message: str
) -> None:
    feeds = {**_feeds(), name: value}
    with p2.load(DYNAMIC_REF) as handle, pytest.raises(p2.P2Error) as error:
        p2.infer(handle, feeds)
    assert error.value.code == ErrorCode.INVALID_INPUT_SHAPE
    assert message in error.value.detail


@pytest.mark.usefixtures("dynamic_engine")
def test_non_contiguous_input_and_scalar_stay_valid() -> None:
    wide = np.ones((2, FEATURES * 2), dtype=np.float32)
    feeds = {**_feeds(), "x": wide[:, ::2]}
    with p2.load(DYNAMIC_REF) as handle:
        outputs, _ = p2.infer(handle, feeds)
    assert outputs["y"].shape == (2, FEATURES)


def test_static_float32_model_keeps_manifest_v1(synth_engine: Path) -> None:
    raw: object = json.loads((synth_engine / "manifest.json").read_text("utf-8"))
    assert isinstance(raw, dict)
    assert raw["schema_version"] == STATIC_SCHEMA_VERSION
    assert "profile" not in raw["inputs"]["images"]
    assert raw["max_batch"] == 1


@pytest.mark.usefixtures("synth_engine")
def test_handle_keeps_public_methods_of_0_1_0() -> None:
    with p2.load(ENGINE_REF) as handle:
        assert handle.input_shapes() == {INPUT_NAME: INPUT_SHAPE}
        assert handle.engine_ref == ENGINE_REF
        assert handle.memory_mb == 0.0
        assert handle.released is False


def test_manifest_v1_written_by_0_1_0_still_loads(synth_engine: Path) -> None:
    manifest = read_manifest(synth_engine)
    assert manifest["schema_version"] == STATIC_SCHEMA_VERSION
    with p2.load(synth_engine.name) as handle:
        (shape,) = p2.info(handle)["input_shape"].values()
        outputs, _ = p2.infer(handle, np.zeros(shape, dtype=np.float32))
    assert outputs


@pytest.mark.parametrize(
    ("shape", "dtype", "expected"),
    [
        ([1, 3], "float32", STATIC_SCHEMA_VERSION),
        ([-1, 3], "float32", SCHEMA_VERSION),
        ([1, 3], "int64", SCHEMA_VERSION),
        ([], "float32", SCHEMA_VERSION),
    ],
)
def test_manifest_schema_choice(shape: list[int], dtype: str, expected: int) -> None:
    spec: TensorSpec = {"shape": shape, "dtype": dtype}
    assert manifest_schema({"a": spec}, {"b": {"shape": [1], "dtype": "float32"}}) == expected


def test_max_batch_uses_largest_first_dimension() -> None:
    inputs: dict[str, TensorSpec] = {
        "a": {"shape": [2, 3], "dtype": "float32"},
        "b": {
            "shape": [-1, 3],
            "dtype": "float32",
            "profile": {"min": [1, 3], "opt": [2, 3], "max": [6, 3]},
        },
        "c": {"shape": [], "dtype": "int64"},
    }
    assert max_batch(inputs) == 6  # noqa: PLR2004
    assert max_batch({"c": inputs["c"]}) == 1


def _v2_manifest(tensor: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "engine_ref": "a-1",
        "source": {"onnx_file": "a.onnx", "sha256": "0", "metadata_file": "metadata.json"},
        "inputs": {"x": tensor},
        "outputs": {"y": {"shape": [-1], "dtype": "float32"}},
        "max_batch": 1,
        "variants": {},
        "platform": {},
        "build": {"tool": "t", "built_at": "n", "tf32": False, "workspace_mb": 1},
    }


@pytest.mark.parametrize(
    ("tensor", "message"),
    [
        ({"shape": [-1, 3], "dtype": "float32"}, "sem 'profile'"),
        ({"shape": [1], "dtype": "float16"}, "float16"),
        ({"shape": [-2], "dtype": "float32"}, "inteiros positivos"),
        (
            {
                "shape": [-1, 3],
                "dtype": "float32",
                "profile": {"min": [4, 3], "opt": [2, 3], "max": [8, 3]},
            },
            "min <= opt <= max",
        ),
        (
            {
                "shape": [-1, 3],
                "dtype": "float32",
                "profile": {"min": [1, 4], "opt": [2, 4], "max": [8, 4]},
            },
            "fixo",
        ),
        (
            {"shape": [-1, 3], "dtype": "float32", "profile": {"min": [1], "opt": [1], "max": [1]}},
            "eixos",
        ),
    ],
)
def test_parse_manifest_v2_rejects(tensor: Mapping[str, object], message: str) -> None:
    with pytest.raises(ManifestFormatError, match=message):
        parse_manifest(_v2_manifest(tensor))


def test_manifest_v1_rejects_dynamic_axis() -> None:
    raw = _v2_manifest({"shape": [-1], "dtype": "float32"})
    raw["schema_version"] = STATIC_SCHEMA_VERSION
    with pytest.raises(ManifestFormatError, match="inteiros positivos"):
        parse_manifest(raw)


def test_package_without_task_is_detect(tmp_path: Path) -> None:
    assert read_package(write_package(tmp_path)).task is Task.DETECT


def test_invalid_task(tmp_path: Path) -> None:
    root = write_package(tmp_path, PackageSpec(metadata_overrides={"task": "voz"}))
    with pytest.raises(BuildError, match="campo 'task' inválido"):
        read_package(root)


def test_detect_still_requires_classes(tmp_path: Path) -> None:
    root = write_package(tmp_path, PackageSpec(metadata_drop=("classes",)))
    with pytest.raises(BuildError, match="campo 'classes' ausente"):
        read_package(root)


@pytest.mark.parametrize("task", [task for task in Task if task is not Task.DETECT])
def test_other_tasks_skip_detection_checks(tmp_path: Path, task: Task) -> None:
    spec = PackageSpec(
        output_rows=9,
        extra_output=True,
        metadata_overrides={"task": task.value},
        metadata_drop=("classes",),
    )
    package = read_package(write_package(tmp_path, spec))
    assert package.classes == ()
    assert len(validate_package(package).outputs) == 2  # noqa: PLR2004


def test_list_input_shape_needs_single_input(tmp_path: Path) -> None:
    root = write_dynamic_package(tmp_path, {"input_shape": [1, 3]})
    with pytest.raises(BuildError, match="exige 1 entrada"):
        validate_package(read_package(root))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"input_shape": {"x": [-1, FEATURES]}}, "input_shape cita"),
        ({"input_shape": {**INPUT_SHAPES, "x": [2, FEATURES]}}, "input_shape de 'x' diferente"),
        ({"dynamic_shapes": {"x": DYNAMIC_SHAPES["x"]}}, "entrada 'tokens' tem eixo dinâmico"),
        ({"dynamic_shapes": {**DYNAMIC_SHAPES, "nada": DYNAMIC_SHAPES["x"]}}, "cita 'nada'"),
        (
            {"dynamic_shapes": {**DYNAMIC_SHAPES, "sr": {"min": [], "opt": [], "max": []}}},
            "não tem eixo dinâmico",
        ),
        (
            {
                "dynamic_shapes": {
                    **DYNAMIC_SHAPES,
                    "x": {"min": [4, 4], "opt": [2, 4], "max": [8, 4]},
                }
            },
            "min <= opt <= max",
        ),
        (
            {
                "dynamic_shapes": {
                    **DYNAMIC_SHAPES,
                    "x": {"min": [1, 5], "opt": [2, 5], "max": [8, 5]},
                }
            },
            "fixo em 4",
        ),
    ],
)
def test_package_validation(tmp_path: Path, overrides: Mapping[str, object], message: str) -> None:
    root = write_dynamic_package(tmp_path, overrides)
    with pytest.raises(BuildError, match=message):
        validate_package(read_package(root))


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("x", "deve ser um objeto"),
        ({"x": {"min": [1, 4]}}, "exatamente min, opt e max"),
        ({"x": {"min": [0, 4], "opt": [1, 4], "max": [1, 4]}}, "inteiros positivos"),
    ],
)
def test_dynamic_shapes_format(tmp_path: Path, value: object, message: str) -> None:
    root = write_dynamic_package(tmp_path, {"dynamic_shapes": value})
    with pytest.raises(BuildError, match=message):
        read_package(root)


def test_unsupported_dtype_rejected(tmp_path: Path) -> None:
    write_dynamic_onnx(tmp_path / ONNX_NAME, input_type=TensorProto.FLOAT16)
    write_metadata(tmp_path)
    with pytest.raises(BuildError, match="tipo float16 de 'x' não suportado"):
        validate_package(read_package(tmp_path))


def test_external_weights_rejected(tmp_path: Path) -> None:
    weight = numpy_helper.from_array(np.ones(1024, dtype=np.float32), name="w")
    graph = helper.make_graph(
        [helper.make_node("Add", ["a", "w"], ["b"])],
        "ext",
        [helper.make_tensor_value_info("a", TensorProto.FLOAT, [1024])],
        [helper.make_tensor_value_info("b", TensorProto.FLOAT, [1024])],
        [weight],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", ONNX_OPSET)], ir_version=ONNX_IR_VERSION
    )
    onnx.save(model, str(tmp_path / ONNX_NAME), save_as_external_data=True, size_threshold=0)
    write_metadata(tmp_path, {"input_shape": [1024]}, drop=("dynamic_shapes",))
    with pytest.raises(BuildError, match="pesos externos"):
        validate_package(read_package(tmp_path))


def test_io_matches_accepts_engine_resolving_output_axis() -> None:
    inputs: dict[str, TensorSpec] = {"a": {"shape": [-1], "dtype": "int64"}}
    onnx_io = OnnxIO(inputs=inputs, outputs={"b": {"shape": [-1, 8], "dtype": "float32"}})
    assert io_matches(onnx_io, OnnxIO(inputs, {"b": {"shape": [4, -1], "dtype": "float32"}}))
    assert not io_matches(onnx_io, OnnxIO(inputs, {"b": {"shape": [4, 9], "dtype": "float32"}}))
    assert not io_matches(onnx_io, OnnxIO(inputs, {"b": {"shape": [4, 8], "dtype": "int64"}}))
    assert not io_matches(onnx_io, OnnxIO(inputs, {"b": {"shape": [8], "dtype": "float32"}}))
    assert not io_matches(onnx_io, OnnxIO(inputs, {"c": {"shape": [4, 8], "dtype": "float32"}}))


def test_io_matches_accepts_input_axis_pinned_by_profile() -> None:
    outputs: dict[str, TensorSpec] = {"b": {"shape": [1], "dtype": "float32"}}
    onnx_io = OnnxIO({"a": {"shape": [-1, -1], "dtype": "float32"}}, outputs)
    assert io_matches(onnx_io, OnnxIO({"a": {"shape": [1, 80], "dtype": "float32"}}, outputs))
    assert not io_matches(onnx_io, OnnxIO({"a": {"shape": [1, 80], "dtype": "int64"}}, outputs))


def test_pin_fixed_axes_only_where_min_equals_max(tmp_path: Path) -> None:
    write_dynamic_onnx(tmp_path / ONNX_NAME)
    profiles: dict[str, ShapeProfile] = {
        "x": {"min": [1, FEATURES], "opt": [2, FEATURES], "max": [MAX_BATCH, FEATURES]},
        "tokens": {"min": [1, 5], "opt": [1, 5], "max": [1, 5]},
    }
    pinned = tmp_path / "pinned.onnx"
    pinned.write_bytes(pin_fixed_axes((tmp_path / ONNX_NAME).read_bytes(), profiles))
    io = read_onnx_io(pinned)
    assert io.inputs["x"]["shape"] == [-1, FEATURES]
    assert io.inputs["tokens"]["shape"] == [1, 5]
    assert io.inputs["sr"]["shape"] == []


def test_pin_fixed_axes_keeps_static_model_bytes() -> None:
    assert pin_fixed_axes(b"sem perfil", {}) == b"sem perfil"


def test_bench_inputs_use_opt_shape_and_dtype(dynamic_engine: Path) -> None:
    inputs = bench_inputs(read_manifest(dynamic_engine)["inputs"])
    assert inputs["x"].shape == tuple(DYNAMIC_SHAPES["x"]["opt"])
    assert inputs["x"].dtype == np.float32
    assert inputs["tokens"].shape == tuple(DYNAMIC_SHAPES["tokens"]["opt"])
    assert inputs["tokens"].dtype == np.int64
    assert inputs["sr"].shape == ()
