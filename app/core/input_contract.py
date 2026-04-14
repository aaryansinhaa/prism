"""Input contract validation helpers.

Supports two modes based on stored `expected_input_json`:
1) Schema-like mode (subset of JSON Schema)
2) Example mode (payload must match example shape and primitive types)
"""

from __future__ import annotations

import json
from typing import Any

_SCHEMA_KEYS = {
    "$schema",
    "type",
    "properties",
    "required",
    "items",
    "enum",
    "additionalProperties",
    "minimum",
    "maximum",
}


def _is_schema_like(contract: Any) -> bool:
    return isinstance(contract, dict) and any(key in contract for key in _SCHEMA_KEYS)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _type_ok(expected_type: str, value: Any) -> bool:
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "number":
        return _is_number(value)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "null":
        return value is None
    return False


def _validate_schema(schema: dict[str, Any], value: Any, path: str = "$") -> str | None:
    expected_type = schema.get("type")
    if isinstance(expected_type, str) and not _type_ok(expected_type, value):
        return f"{path}: expected type '{expected_type}'"

    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and value not in enum_values:
        return f"{path}: value not in allowed enum"

    if (
        expected_type == "object" or "properties" in schema or "required" in schema
    ) and not isinstance(value, dict):
        return f"{path}: expected object"

    if isinstance(value, dict):
        required = schema.get("required", [])
        if isinstance(required, list):
            for key in required:
                if key not in value:
                    return f"{path}.{key}: required field missing"

        properties = schema.get("properties", {})
        if isinstance(properties, dict):
            for key, sub_schema in properties.items():
                if key in value and isinstance(sub_schema, dict):
                    error = _validate_schema(sub_schema, value[key], f"{path}.{key}")
                    if error:
                        return error

        additional = schema.get("additionalProperties", True)
        if additional is False and isinstance(properties, dict):
            allowed = set(properties.keys())
            for key in value:
                if key not in allowed:
                    return f"{path}.{key}: additional properties not allowed"

    if (expected_type == "array" or "items" in schema) and not isinstance(value, list):
        return f"{path}: expected array"

    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        item_schema = schema["items"]
        for index, item in enumerate(value):
            error = _validate_schema(item_schema, item, f"{path}[{index}]")
            if error:
                return error

    if (
        "minimum" in schema
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    ):
        numeric_value = float(value)
        minimum = schema.get("minimum")
        if isinstance(minimum, (int, float)) and numeric_value < float(minimum):
            return f"{path}: value must be >= {minimum}"

    if (
        "maximum" in schema
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    ):
        numeric_value = float(value)
        maximum = schema.get("maximum")
        if isinstance(maximum, (int, float)) and numeric_value > float(maximum):
            return f"{path}: value must be <= {maximum}"

    return None


def _validate_example(example: Any, value: Any, path: str = "$") -> str | None:
    if isinstance(example, dict):
        if not isinstance(value, dict):
            return f"{path}: expected object"
        for key, expected_sub in example.items():
            if key not in value:
                return f"{path}.{key}: required field missing"
            error = _validate_example(expected_sub, value[key], f"{path}.{key}")
            if error:
                return error
        return None

    if isinstance(example, list):
        if not isinstance(value, list):
            return f"{path}: expected array"
        if not example:
            return None
        sample = example[0]
        for index, item in enumerate(value):
            error = _validate_example(sample, item, f"{path}[{index}]")
            if error:
                return error
        return None

    if isinstance(example, bool):
        return None if isinstance(value, bool) else f"{path}: expected boolean"

    if example is None:
        return None if value is None else f"{path}: expected null"

    if isinstance(example, int) and not isinstance(example, bool):
        return (
            None
            if isinstance(value, int) and not isinstance(value, bool)
            else f"{path}: expected integer"
        )

    if isinstance(example, float):
        return None if _is_number(value) else f"{path}: expected number"

    if isinstance(example, str):
        return None if isinstance(value, str) else f"{path}: expected string"

    return None


def validate_payload_against_expected_input_json(
    expected_input_json: str | None,
    payload: Any,
) -> tuple[bool, str | None]:
    """Validate payload against stored expected input JSON contract.

    Returns:
      (True, None) when valid or no contract.
      (False, reason) when invalid.
    """
    if not expected_input_json:
        return True, None

    try:
        contract = json.loads(expected_input_json)
    except json.JSONDecodeError:
        return False, "Configured expected input contract is invalid JSON"

    if _is_schema_like(contract):
        if not isinstance(contract, dict):
            return False, "Schema contract must be an object"
        error = _validate_schema(contract, payload)
    else:
        error = _validate_example(contract, payload)

    if error:
        return False, error
    return True, None
