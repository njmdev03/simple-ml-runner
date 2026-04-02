from dataclasses import is_dataclass, fields

def build_dataclass(cls, data: dict):
    kwargs = {}

    for f in fields(cls):
        if f.name not in data:
            continue

        value = data[f.name]
        field_type = f.type

        if is_dataclass(field_type):
            value = build_dataclass(field_type, value)

        kwargs[f.name] = value

    return cls(**kwargs)