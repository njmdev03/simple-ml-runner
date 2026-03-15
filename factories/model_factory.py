from registries import ModelRegistry


def resolve_model(cfg: dict):
    key = next(iter(cfg))
    params = cfg[key]

    cls = ModelRegistry.get(key)
    return cls(**params)  # PythonModel or any built-in model
