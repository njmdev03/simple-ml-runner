from dataclasses import is_dataclass, fields, MISSING
from typing import get_origin, get_args

from simple_config.schema import Variant


class ConfigBuilder:
    """Builder for constructing dataclass-based configurations from dictionaries.

    Attributes:
        schema: The root dataclass schema to build.
    """

    def __init__(self, schema):
        """Initialize builder.

        Args:
            schema: A dataclass class to use as the template.
        """
        self.schema = schema

    def build(self, data: dict):
        """Build configuration from dictionary.

        Args:
            data: Raw configuration data.

        Returns:
            An instance of the schema dataclass.
        """
        return _build_dataclass(self.schema, data)

def _build_dataclass(schema, data: dict):
    """Recursively build a dataclass from a dictionary.

    Args:
        schema: The dataclass class to instantiate.
        data: Data to populate fields.

    Returns:
        Populated dataclass instance.

    Raises:
        MissingFieldException: Required field missing in data.
    """
    kwargs = {}
    for f in fields(schema):
        key = f.name
        if key in data:
            kwargs[key] = _convert_value(f.type, data[key], field=f)
        elif f.default is not MISSING:
            kwargs[key] = f.default
        elif f.default_factory is not MISSING:
            kwargs[key] = f.default_factory()
        else:
            raise MissingFieldException(key, schema.__name__)

    return schema(**kwargs)

def _convert_value(type_hint, value, field=None):
    """Convert raw value to expected type based on type hint.

    Args:
        type_hint: Expected type (primitive, dataclass, List, Dict, or Variant).
        value: Input value to convert.
        field: Optional dataclass field metadata.

    Returns:
        Converted value matching type_hint.

    Raises:
        ConfigTypeError: Input value type does not match type_hint.
    """
    field_name = field.name if field else "?"
    if type_hint is Variant:
        return _convert_variant(field, value)

    if is_dataclass(type_hint):
        if not isinstance(value, dict):
            raise ConfigTypeError(field_name, "dict", value)
        return _build_dataclass(type_hint, value)

    origin = get_origin(type_hint)
    if origin is list:
        if not isinstance(value, list):
            raise ConfigTypeError(field_name, "list", value)
        inner = get_args(type_hint)[0]
        return [_convert_value(inner, v) for v in value]

    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigTypeError(field_name, "dict", value)
        kt, vt = get_args(type_hint)
        return {
            _convert_value(kt, k): _convert_value(vt, v)
            for k, v in value.items()
        }

    check_type = origin or type_hint
    if check_type is not None and not isinstance(value, check_type):
        raise ConfigTypeError(field_name, check_type, value)

    return value

def _convert_variant(field, value):
    """Handle conversion for Variant types.

    Args:
        field: Dataclass field containing Variant definition.
        value: Input value (string, dict, or empty).

    Returns:
        Instantiated dataclass for the selected variant.

    Raises:
        VariantFormatException: Input doesn't match string or single-key dict format.
        ValueError: Variant name not found in mapping.
    """
    v_def = field.default
    if isinstance(value, str):
        name, payload = value, {}
    elif isinstance(value, dict):
        if len(value) == 1:
            name, payload = next(iter(value.items()))
        elif not value and v_def.default:
            name, payload = v_def.default, {}
        else:
            raise VariantFormatException(field.name, value)
    else:
        raise VariantFormatException(field.name, value)

    if name not in v_def.mapping:
        options = ", ".join(v_def.mapping.keys())
        raise ValueError(f"Unknown variant '{name}' for field '{field.name}'. Expected one of: {options}")

    return _build_dataclass(v_def.mapping[name], payload)

class ConfigBuilderException(Exception):
    """Base class for config builder errors."""
    pass

class MissingFieldException(ConfigBuilderException):
    """Raised when a required dataclass field is missing from input data."""
    def __init__(self, field_name: str, schema_name: str):
        super().__init__(f"Missing required field '{field_name}' in dataclass '{schema_name}'")

class VariantFormatException(ConfigBuilderException):
    """Raised when Variant data format is invalid."""
    def __init__(self, field_name: str, value):
        super().__init__(f"Invalid format for variant field '{field_name}': {value!r}. Expected string or single-key dict.")

class ConfigTypeError(ConfigBuilderException):
    """Raised when input value type does not match schema definition."""
    def __init__(self, field_name: str, expected_type, actual_value):
        actual_type = type(actual_value).__name__
        super().__init__(f"Type mismatch for field '{field_name}': expected {expected_type}, got {actual_type} ({actual_value!r})")
