import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from helpers.model_package import ONNX_IR_VERSION, ONNX_OPSET
from p2.build.output import BuildStamp, write_cpu_only
from p2.build.package import read_package, validate_package

ARTIFACT_ID: Final = "dyn"
ARTIFACT_VERSION: Final = "1.0.0"
DYNAMIC_REF: Final = f"{ARTIFACT_ID}-{ARTIFACT_VERSION}"
ONNX_NAME: Final = "dyn.onnx"
FEATURES: Final = 4
MAX_BATCH: Final = 8
MAX_TOKENS: Final = 16
DYNAMIC_SHAPES: Final[Mapping[str, Mapping[str, list[int]]]] = {
    "x": {"min": [1, FEATURES], "opt": [2, FEATURES], "max": [MAX_BATCH, FEATURES]},
    "tokens": {"min": [1, 1], "opt": [1, 4], "max": [1, MAX_TOKENS]},
}
INPUT_SHAPES: Final[Mapping[str, list[int]]] = {"x": [-1, FEATURES], "tokens": [1, -1], "sr": []}


def write_dynamic_onnx(path: Path, *, input_type: int = TensorProto.FLOAT) -> None:
    nodes = [
        helper.make_node("Constant", [], ["two"], value=_scalar(2.0)),
        helper.make_node("Constant", [], ["zero"], value=_scalar(0.0)),
        helper.make_node("Mul", ["x", "two"], ["y"]),
        helper.make_node("Greater", ["x", "zero"], ["mask"]),
        helper.make_node("Add", ["tokens", "sr"], ["next"]),
    ]
    inputs = [
        helper.make_tensor_value_info("x", input_type, ["batch", FEATURES]),
        helper.make_tensor_value_info("tokens", TensorProto.INT64, [1, "seq"]),
        helper.make_tensor_value_info("sr", TensorProto.INT64, []),
    ]
    outputs = [
        helper.make_tensor_value_info("y", input_type, ["batch", FEATURES]),
        helper.make_tensor_value_info("mask", TensorProto.BOOL, ["batch", FEATURES]),
        helper.make_tensor_value_info("next", TensorProto.INT64, [1, "seq"]),
    ]
    graph = helper.make_graph(nodes, "dyn", inputs, outputs)
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", ONNX_OPSET)],
        ir_version=ONNX_IR_VERSION,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))


def write_metadata(
    root: Path,
    overrides: Mapping[str, object] | None = None,
    drop: Sequence[str] = (),
) -> Path:
    onnx_path = root / ONNX_NAME
    metadata: dict[str, object] = {
        "artifact_id": ARTIFACT_ID,
        "artifact_version": ARTIFACT_VERSION,
        "onnx_file": ONNX_NAME,
        "hash_sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
        "precision": "FP32",
        "task": "other",
        "input_shape": dict(INPUT_SHAPES),
        "dynamic_shapes": dict(DYNAMIC_SHAPES),
        **(overrides or {}),
    }
    for key in drop:
        metadata.pop(key, None)
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return root


def write_dynamic_package(
    root: Path,
    overrides: Mapping[str, object] | None = None,
    drop: Sequence[str] = (),
) -> Path:
    write_dynamic_onnx(root / ONNX_NAME)
    return write_metadata(root, overrides, drop)


def build_dynamic_engine(package_root: Path, engine_root: Path) -> Path:
    package = read_package(write_dynamic_package(package_root))
    io = validate_package(package)
    stamp = BuildStamp(built_at=datetime.now(timezone.utc), tool_version="test")
    return write_cpu_only(package, io, engine_root=engine_root, force=False, stamp=stamp)


def _scalar(value: float) -> onnx.TensorProto:
    return numpy_helper.from_array(np.array(value, dtype=np.float32))
