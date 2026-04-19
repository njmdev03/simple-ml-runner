from dataclasses import is_dataclass, fields, MISSING
from typing import get_origin, get_args

from simple_config.schema import Variant


class ConfigBuilder:
    def __init__(self, schema):
        self.schema = schema
        # self.registry = registry or {} # , registry=None

    def build(self, data: dict):
        return _build_dataclass(self.schema, data)

def _build_dataclass(schema, data: dict):
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
            raise MissingFieldException()
    return schema(**kwargs)

def _convert_value(type_hint, value, field=None):
    if type_hint is Variant:
        return _convert_variant(field, value)

    if is_dataclass(type_hint):
        return _build_dataclass(type_hint, value)

    origin = get_origin(type_hint)
    if origin is list:
        inner = get_args(type_hint)[0]
        return [_convert_value(inner, v) for v in value]

    if origin is dict:
        kt, vt = get_args(type_hint)
        return {
            _convert_value(kt, k): _convert_value(vt, v) 
            for k, v in value.items()
        }

    if not isinstance(value, type_hint):
        raise TypeError()
    return value

def _convert_variant(field, value):
    v_def = field.default
    if isinstance(value, str):
        name, payload = value, {}
    elif isinstance(value, dict):
        if len(value) == 1:
            name, payload = next(iter(value.items()))
        elif not value and v_def.default:
            name, payload = v_def.default, {}
        else:
            raise VariantFormatException()
    else:
        raise VariantFormatException()

    if name not in v_def.mapping:
        raise ValueError(f"Unknown variant: {name}")

    return _build_dataclass(v_def.mapping[name], payload)

class MissingFieldException(Exception):
    pass

class VariantFormatException(Exception):
    pass
