import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from p2.build.output import BuildStamp, write_cpu_only
from p2.build.package import read_package, validate_package

ARTIFACT_ID: Final = "synth"
ARTIFACT_VERSION: Final = "1.0.0"
ENGINE_REF: Final = f"{ARTIFACT_ID}-{ARTIFACT_VERSION}"
INPUT_NAME: Final = "images"
OUTPUT_NAME: Final = "output0"
INPUT_SHAPE: Final[tuple[int, ...]] = (1, 3, 8, 8)
CLASSES: Final[tuple[str, ...]] = ("person", "helmet", "vest")
ANCHORS: Final = 4
BOX_ROWS: Final = 4
ONNX_NAME: Final = "model.onnx"
ONNX_OPSET: Final = 17
ONNX_IR_VERSION: Final = 10

Dim = int | str


@dataclass(frozen=True, slots=True)
class PackageSpec:
    onnx_input_shape: tuple[Dim, ...] = INPUT_SHAPE
    output_rows: int = BOX_ROWS + len(CLASSES)
    extra_input: bool = False
    extra_output: bool = False
    onnx_name: str = ONNX_NAME
    metadata_overrides: Mapping[str, object] = field(default_factory=dict)
    metadata_drop: tuple[str, ...] = ()
    fails_at_runtime: bool = False


def write_onnx(path: Path, spec: PackageSpec) -> None:
    output_shape = [1, spec.output_rows, ANCHORS]
    nodes = _runtime_failure_nodes() if spec.fails_at_runtime else _output_nodes(output_shape)
    inputs = [_value(INPUT_NAME, spec.onnx_input_shape)]
    outputs = [_value(OUTPUT_NAME, output_shape)]
    if spec.extra_input:
        inputs.append(_value("extra", INPUT_SHAPE))
    if spec.extra_output:
        nodes.append(helper.make_node("Identity", [INPUT_NAME], ["extra_out"]))
        outputs.append(_value("extra_out", spec.onnx_input_shape))
    graph = helper.make_graph(nodes, "synth", inputs, outputs)
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", ONNX_OPSET)],
        ir_version=ONNX_IR_VERSION,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))


def write_package(root: Path, spec: PackageSpec | None = None) -> Path:
    chosen = spec or PackageSpec()
    root.mkdir(parents=True, exist_ok=True)
    onnx_path = root / chosen.onnx_name
    write_onnx(onnx_path, chosen)
    metadata = {**default_metadata(onnx_path, chosen.onnx_name), **chosen.metadata_overrides}
    for key in chosen.metadata_drop:
        metadata.pop(key, None)
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return root


def build_engine(package_root: Path, engine_root: Path, spec: PackageSpec | None = None) -> Path:
    package = read_package(write_package(package_root, spec))
    io = validate_package(package)
    stamp = BuildStamp(built_at=datetime.now(timezone.utc), tool_version="test")
    return write_cpu_only(package, io, engine_root=engine_root, force=False, stamp=stamp)


def default_metadata(onnx_path: Path, onnx_name: str) -> dict[str, object]:
    return {
        "artifact_id": ARTIFACT_ID,
        "artifact_version": ARTIFACT_VERSION,
        "onnx_file": onnx_name,
        "hash_sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
        "input_shape": list(INPUT_SHAPE),
        "classes": list(CLASSES),
        "precision": "FP32",
    }


def _output_nodes(output_shape: Sequence[int]) -> list[onnx.NodeProto]:
    base = numpy_helper.from_array(np.arange(np.prod(output_shape), dtype=np.float32))
    return [
        helper.make_node("ReduceMean", [INPUT_NAME], ["mean"], keepdims=1),
        helper.make_node("Constant", [], ["scalar_shape"], value=_int64([1])),
        helper.make_node("Reshape", ["mean", "scalar_shape"], ["mean_flat"]),
        helper.make_node("Constant", [], ["base"], value=base),
        helper.make_node("Constant", [], ["shape"], value=_int64(output_shape)),
        helper.make_node("Reshape", ["base", "shape"], ["base_shaped"]),
        helper.make_node("Add", ["base_shaped", "mean_flat"], [OUTPUT_NAME]),
    ]


def _runtime_failure_nodes() -> list[onnx.NodeProto]:
    return [
        helper.make_node("Constant", [], ["flat"], value=_int64([-1])),
        helper.make_node("Reshape", [INPUT_NAME, "flat"], ["values"]),
        helper.make_node("ReduceMax", ["values"], ["peak"], axes=[0], keepdims=1),
        helper.make_node("Cast", ["peak"], ["target"], to=TensorProto.INT64),
        helper.make_node("Reshape", [INPUT_NAME, "target"], [OUTPUT_NAME]),
    ]


def _int64(values: Sequence[int]) -> onnx.TensorProto:
    return numpy_helper.from_array(np.array(values, dtype=np.int64))


def _value(name: str, shape: Sequence[Dim]) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, list(shape))
