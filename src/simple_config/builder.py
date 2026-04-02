from typing import Type, Any, Dict, List, Union, get_origin, get_args
from dataclasses import fields, is_dataclass, MISSING

from simple_config.config_registry import ConfigRegistry
from simple_config.loader import load_raw_config
from simple_config.merger import merge_dicts
from simple_config.exceptions import SimpleConfigError, ConfigNotFoundError, ConfigValidationError, ConfigTypeError

def build_config(name: str, files: Union[str, List[str]]) -> Any:
    """Load, merge, and build a typed config dataclass.

    Args:
        name: The name of the root config schema to build.
        files: A single path or list of paths to configuration files.

    Returns:
        An instance of the registered dataclass.

    Raises:
        ConfigNotFoundError: If the schema name or a config file is not found.
        ConfigValidationError: If the configuration data is invalid for the schema.
        ConfigTypeError: If there's a type mismatch in the configuration.
        SimpleConfigError: For other general configuration errors.
    """
    try:
        if isinstance(files, str):
            files = [files]

        try:
            cfg_cls = ConfigRegistry.get(name)
        except (ConfigNotFoundError, KeyError) as e:
            if isinstance(e, KeyError):
                raise ConfigNotFoundError(f"Config schema '{name}' not found.") from e
            raise

        merged_data: Dict[str, Any] = {}

        for f in files:
            data = load_raw_config(f)
            merged_data = merge_dicts(merged_data, data)

        config = _instantiate_dataclass(cfg_cls, merged_data)

        return config
    except SimpleConfigError:
        raise
    except Exception as e:
        raise SimpleConfigError(f"Error building config '{name}': {str(e)}") from e

def _instantiate_dataclass(cls: Type, data: Any, path: str = "") -> Any:
    """Recursively instantiate dataclass, supporting nested dataclasses and polymorphic types.

    Args:
        cls: The type to instantiate.
        data: The raw data (usually a dict) to use for instantiation.
        path: The current path in the configuration hierarchy for error reporting.

    Returns:
        The instantiated object.

    Raises:
        ConfigValidationError: If required fields are missing.
        ConfigTypeError: If there's a type mismatch.
    """
    # Handle Polymorphic Types (Friendly Schema)
    mapping = ConfigRegistry.get_mapping(cls)
    if mapping:
        if not isinstance(data, dict):
            raise ConfigTypeError(f"At '{path}': Expected a dict for polymorphic type '{cls.__name__}', got '{type(data).__name__}'")

        if len(data) != 1:
            raise ConfigValidationError(f"At '{path}': Friendly Schema for '{cls.__name__}' must have exactly one key (the implementation name), but found {list(data.keys())}")

        impl_name = next(iter(data))
        impl_data = data[impl_name]

        if impl_name not in mapping:
            raise ConfigNotFoundError(f"At '{path}': No implementation named '{impl_name}' registered for '{cls.__name__}'. Registered: {list(mapping.keys())}")

        impl_cls = mapping[impl_name]
        return _instantiate_dataclass(impl_cls, impl_data, path=f"{path}.{impl_name}" if path else impl_name)

    # Handle Dataclasses
    if is_dataclass(cls):
        if not isinstance(data, dict):
            raise ConfigTypeError(f"At '{path}': Expected a dict for dataclass '{cls.__name__}', but got '{type(data).__name__}'")

        init_values = {}
        for field in fields(cls):
            field_path = f"{path}.{field.name}" if path else field.name

            if field.name in data:
                value = data[field.name]
                init_values[field.name] = _handle_type(field.type, value, field_path)
            elif field.default is not MISSING:
                init_values[field.name] = field.default
            elif field.default_factory is not MISSING:
                init_values[field.name] = field.default_factory()
            else:
                raise ConfigValidationError(f"At '{path}': Field '{field.name}' is required in '{cls.__name__}' but was not provided.")

        return cls(**init_values)

    # 3. Handle base types (should be handled by _handle_type, but as a fallback)
    return _handle_type(cls, data, path)

def _handle_type(type_hint: Any, value: Any, path: str) -> Any:
    """Handle type conversion and validation for various type hints.

    Args:
        type_hint: The type hint to validate against.
        value: The value to validate/convert.
        path: The path for error reporting.

    Returns:
        The validated/converted value.
    """
    origin = get_origin(type_hint)
    args = get_args(type_hint)

    # Handle Union (including Optional)
    if origin is Union:
        # Check if any of the types in the Union match
        for arg in args:
            try:
                # If it's None and NoneType is in Union, it's valid
                if value is None and arg is type(None):
                    return None
                return _handle_type(arg, value, path)
            except (ConfigTypeError, ConfigValidationError):
                continue
        raise ConfigTypeError(f"At '{path}': Value '{value}' does not match any type in {type_hint}")

    # Handle List
    if origin is list:
        if not isinstance(value, list):
            raise ConfigTypeError(f"At '{path}': Expected a list, but got '{type(value).__name__}'")
        item_type = args[0] if args else Any
        return [_handle_type(item_type, item, f"{path}[{i}]") for i, item in enumerate(value)]

    # Handle Dict
    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigTypeError(f"At '{path}': Expected a dict, but got '{type(value).__name__}'")
        key_type = args[0] if args else Any
        val_type = args[1] if args else Any
        return {
            _handle_type(key_type, k, f"{path}.<key>"): _handle_type(val_type, v, f"{path}.{k}")
            for k, v in value.items()
        }

    # Handle nested dataclasses or polymorphic types
    if is_dataclass(type_hint) or ConfigRegistry.get_mapping(type_hint):
        return _instantiate_dataclass(type_hint, value, path)

    # Handle basic types
    if isinstance(type_hint, type):
        # Special case: float can accept int
        if type_hint is float and isinstance(value, int):
            return float(value)

        if value is not None and not isinstance(value, type_hint):
            raise ConfigTypeError(f"At '{path}': Expected type '{type_hint.__name__}', but got '{type(value).__name__}'")
        return value

    # Fallback for Any or unhandled hints
    return value
