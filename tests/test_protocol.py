"""Contract test: src/robot3d/protocol.py must describe exactly the JSON that
web/src/protocol.ts (the single source of truth) describes.

How: ts-json-schema-generator (an npm dev dependency) turns the TypeScript
types into a JSON Schema, and Pydantic produces one for the Python models.
The two tools phrase schemas differently, so we reduce both to the same
simple "shape" (field names, which are required, value types) and compare
message by message. Descriptions and titles are ignored.
"""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from pydantic import TypeAdapter

from robot3d.protocol import (
    ClientMessage,
    EvaluateResponse,
    RunDetail,
    RunSummary,
    ScalarsResponse,
    ServerMessage,
)

WEB_DIR = Path(__file__).resolve().parents[1] / "web"
GENERATOR_PKG = WEB_DIR / "node_modules" / "ts-json-schema-generator" / "package.json"


@pytest.fixture(scope="module")
def ts_schema() -> dict:
    node = shutil.which("node")
    if node is None or not GENERATOR_PKG.is_file():
        pytest.skip("needs Node.js and `npm install` in web/")
    entry = GENERATOR_PKG.parent / json.loads(GENERATOR_PKG.read_text())["bin"]["ts-json-schema-generator"]
    result = subprocess.run(
        [node, str(entry), "--path", "src/protocol.ts", "--type", "*", "--tsconfig", "tsconfig.json"],
        cwd=WEB_DIR,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"ts-json-schema-generator failed:\n{result.stderr}")
    return json.loads(result.stdout)


def shape(node: dict, root: dict) -> Any:
    """Reduce a JSON Schema node to a comparable value, ignoring wording."""
    while "$ref" in node:  # "#/definitions/X" (TypeScript) or "#/$defs/X" (Pydantic)
        node = _resolve(root, node["$ref"])
    if "const" in node:
        return ("const", node["const"])
    if "enum" in node:
        values = node["enum"]
        return ("const", values[0]) if len(values) == 1 else ("enum", tuple(sorted(values)))
    for key in ("anyOf", "oneOf"):
        if key in node:
            return ("union", tuple(sorted((shape(n, root) for n in node[key]), key=repr)))
    kind = node.get("type")
    if isinstance(kind, list):  # {"type": ["number", "null"]} (TypeScript) = anyOf (Pydantic)
        return ("union", tuple(sorted((shape({**node, "type": k}, root) for k in kind), key=repr)))
    if kind == "object":
        values = node.get("additionalProperties")
        if isinstance(values, dict) and not node.get("properties"):
            return ("map", shape(values, root))  # Record<string, T> / dict[str, T]
        fields = tuple(sorted((name, shape(n, root)) for name, n in node.get("properties", {}).items()))
        return ("object", fields, tuple(sorted(node.get("required", []))))
    if kind == "array":
        # A fixed-length list like [number, number, number] can be written as a
        # tuple (Pydantic: prefixItems; old style: items=[...]) or as
        # items + minItems == maxItems (ts-json-schema-generator). Treat those
        # the same: ("array", item shape, length).
        items = node.get("prefixItems", node.get("items"))
        if isinstance(items, list):
            item_shapes = tuple(shape(n, root) for n in items)
            if len(set(item_shapes)) == 1:
                return ("array", item_shapes[0], len(item_shapes))
            return ("tuple", item_shapes)
        fixed = node.get("minItems") if node.get("minItems") == node.get("maxItems") else None
        return ("array", shape(items, root), fixed)
    if kind in ("number", "integer"):
        return ("number",)  # TypeScript has a single number type
    if kind in ("string", "boolean", "null"):
        return (kind,)
    raise AssertionError(f"shape() doesn't understand this schema node: {node}")


def _resolve(root: dict, ref: str) -> dict:
    node = root
    for part in ref.removeprefix("#/").split("/"):
        node = node[part]
    return node


def by_message_type(union_shape: Any) -> dict[str, Any]:
    """{"scene": <shape of SceneMessage>, ...} from the shape of a message union."""
    kind, variants = union_shape
    assert kind == "union"
    result = {}
    for variant in variants:
        fields = dict(variant[1])
        result[fields["type"][1]] = variant
    return result


@pytest.mark.parametrize(
    "name, python_type",
    [("ServerMessage", ServerMessage), ("ClientMessage", ClientMessage)],
)
def test_python_mirror_matches_typescript(ts_schema, name, python_type):
    ts = by_message_type(shape(ts_schema["definitions"][name], ts_schema))
    py_schema = TypeAdapter(python_type).json_schema(mode="serialization")
    py = by_message_type(shape(py_schema, py_schema))

    assert sorted(py) == sorted(ts), f"{name}: message types differ (Python vs TypeScript)"
    for msg_type in ts:
        ts_fields, ts_required = dict(ts[msg_type][1]), ts[msg_type][2]
        py_fields, py_required = dict(py[msg_type][1]), py[msg_type][2]
        assert sorted(py_fields) == sorted(ts_fields), f"'{msg_type}' message: field names differ"
        for field in ts_fields:
            assert py_fields[field] == ts_fields[field], f"'{msg_type}.{field}': types differ"
        assert py_required == ts_required, f"'{msg_type}' message: required fields differ"


@pytest.mark.parametrize("model", [RunSummary, RunDetail, ScalarsResponse, EvaluateResponse], ids=lambda m: m.__name__)
def test_http_api_types_match_typescript(ts_schema, model):
    """The dashboard's HTTP responses (same names in both languages;
    RunDetail pulls in its nested types)."""
    ts = shape(ts_schema["definitions"][model.__name__], ts_schema)
    py_schema = model.model_json_schema(mode="serialization")
    py = shape(py_schema, py_schema)
    ts_fields, py_fields = dict(ts[1]), dict(py[1])
    assert sorted(py_fields) == sorted(ts_fields), f"{model.__name__}: field names differ"
    for field in ts_fields:
        assert py_fields[field] == ts_fields[field], f"{model.__name__}.{field}: types differ"
    assert py == ts


def test_shape_catches_a_mismatch(ts_schema):
    """Guard against a check that always passes: a changed field must be detected."""
    scene = ts_schema["definitions"]["SceneMessage"]
    tampered = json.loads(json.dumps(ts_schema))
    tampered["definitions"]["SceneMessage"]["properties"]["timestep"] = {"type": "string"}
    assert shape(scene, ts_schema) != shape(tampered["definitions"]["SceneMessage"], tampered)
