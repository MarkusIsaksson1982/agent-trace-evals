"""Small, dependency-free validator for a supported JSON Schema subset."""
from __future__ import annotations
from typing import Any

_KEYS = {"type", "properties", "required", "enum", "items", "additionalProperties", "minLength", "maxLength"}
_TYPES = {"object", "array", "string", "number", "integer", "boolean", "null"}

def _check_schema(schema: Any) -> None:
    if not isinstance(schema, dict): raise ValueError("schema must be an object")
    unknown = set(schema) - _KEYS
    if unknown: raise ValueError(f"unknown schema keyword: {sorted(unknown)[0]}")
    if "type" in schema and schema["type"] not in _TYPES: raise ValueError(f"unknown schema type: {schema['type']!r}")
    if "enum" in schema and not isinstance(schema["enum"], list): raise ValueError("enum must be a list")
    if "properties" in schema:
        if not isinstance(schema["properties"], dict): raise ValueError("properties must be an object")
        for child in schema["properties"].values(): _check_schema(child)
    if "items" in schema: _check_schema(schema["items"])
    if "required" in schema and (not isinstance(schema["required"], list) or not all(isinstance(n, str) for n in schema["required"])): raise ValueError("required must be a list of strings")
    if "required" in schema and "properties" in schema:
        missing = set(schema["required"]) - set(schema["properties"])
        if missing: raise ValueError(f"required property is not defined: {sorted(missing)[0]}")
    if schema.get("additionalProperties", False) is not False: raise ValueError("additionalProperties must be false")
    for key in ("minLength", "maxLength"):
        if key in schema and (isinstance(schema[key], bool) or not isinstance(schema[key], int) or schema[key] < 0): raise ValueError(f"{key} must be a non-negative integer")

def _type_ok(value: Any, expected: str) -> bool:
    if expected == "null": return value is None
    if expected == "boolean": return isinstance(value, bool)
    if expected == "number": return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "integer": return isinstance(value, int) and not isinstance(value, bool)
    return {"object": isinstance(value, dict), "array": isinstance(value, list), "string": isinstance(value, str)}[expected]

def _validate(value: Any, schema: dict[str, Any], path: str, errors: list[str]) -> None:
    expected = schema.get("type")
    if expected and not _type_ok(value, expected): errors.append(f"{path}: expected {expected}"); return
    if "enum" in schema and not any(type(value) is type(member) and value == member for member in schema["enum"]): errors.append(f"{path}: value is not in enum")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]: errors.append(f"{path}: length below minLength")
        if "maxLength" in schema and len(value) > schema["maxLength"]: errors.append(f"{path}: length above maxLength")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        for name in schema.get("required", []):
            if name not in value: errors.append(f"{path}.{name}: required property is missing")
        for name, child in properties.items():
            if name in value: _validate(value[name], child, f"{path}.{name}", errors)
        if schema.get("additionalProperties") is False:
            for name in value:
                if name not in properties: errors.append(f"{path}.{name}: additional property is not allowed")
    if isinstance(value, list) and "items" in schema:
        for index, item in enumerate(value): _validate(item, schema["items"], f"{path}[{index}]", errors)

def validate(instance: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    _check_schema(schema)
    errors: list[str] = []
    _validate(instance, schema, path, errors)
    return errors
