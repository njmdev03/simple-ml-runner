from typing import Type, Any, Dict, List, Union, get_origin, get_args
from pathlib import Path
from dataclasses import fields, is_dataclass, MISSING

from simple_config.config_registry import ConfigRegistry
from simple_config.loader import load_raw_config
from simple_config.merger import merge_dicts
from simple_config.exceptions import SimpleConfigError, ConfigNotFoundError, ConfigValidationError, ConfigTypeError

def build_config(name: str, files: Union[str, List[str]]) -> Any:
    """
    Load, merge, and build a typed config dataclass.

    1. Loads one or more files (with inheritance via the 'config' key).
    2. Merges them sequentially.
    3. Populates the dataclass from the registry.
    4. Handles extensions if present.
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

        # Step 6: Inject extensions (optional implementation based on registry)
        # We look for any keys in merged_data that are registered as extension configs
        # and were NOT consumed by the main dataclass.
        if hasattr(config, "extensions") and isinstance(config.extensions, dict):
            consumed_fields = {f.name for f in fields(cfg_cls)}
            for key, value in merged_data.items():
                if key not in consumed_fields and key != "config":
                    ext_schema_name = f"extension.{key}"
                    try:
                        ext_cls = ConfigRegistry.get(ext_schema_name)
                        config.extensions[key] = _instantiate_dataclass(ext_cls, value)
                    except ConfigNotFoundError:
                        # If not in registry, just leave as dict or ignore if desired
                        pass
                    except ConfigValidationError as e:
                        raise ConfigValidationError(f"Extension '{key}' validation failed: {e}") from e

        return config
    except SimpleConfigError:
        raise
    except Exception as e:
        raise SimpleConfigError(f"Error building config '{name}': {str(e)}") from e

def _instantiate_dataclass(cls: Type, data: Dict[str, Any]) -> Any:
    """Recursively instantiate dataclass, supporting nested dataclasses."""
    if not is_dataclass(cls):
        return data

    init_values = {}

    for field in fields(cls):
        if field.name in data:
            value = data[field.name]

            # Handle nested dataclasses (recursive building)
            if is_dataclass(field.type):
                if isinstance(value, dict):
                    try:
                        value = _instantiate_dataclass(field.type, value)
                    except (ConfigValidationError, ConfigTypeError) as e:
                        # Provide context for nested errors
                        raise type(e)(f"Field '{field.name}' in '{cls.__name__}' is invalid: {e}") from e
                else:
                    # Field expects a dataclass but got something else (not a dict)
                    raise ConfigTypeError(
                        f"Field '{field.name}' in '{cls.__name__}' expected a dict for dataclass '{field.type.__name__}', but got '{type(value).__name__}'."
                    )

            # Basic type checking for common types (int, float, str, bool)
            elif isinstance(field.type, type):
                # If field.type is a simple type hint (like int, str), check it
                # Special case: allow int for float fields
                is_type_match = isinstance(value, field.type)
                if not is_type_match and field.type is float and isinstance(value, int):
                    is_type_match = True
                    value = float(value)

                if not is_type_match:
                    raise ConfigTypeError(
                        f"Field '{field.name}' in '{cls.__name__}' expected type '{field.type.__name__}', but got '{type(value).__name__}'."
                    )

            init_values[field.name] = value
        elif field.default is not MISSING:
            init_values[field.name] = field.default
        elif field.default_factory is not MISSING:
            init_values[field.name] = field.default_factory()
        else:
            raise ConfigValidationError(f"Field '{field.name}' is required in '{cls.__name__}' but was not provided.")

    return cls(**init_values)