def resolve_component(cfg: dict, registry) -> object:
    """
    Generic factory to resolve any component from a registry.

    cfg example for models:
        {"MLP": {"input_dim": 784, "layers": [256, 128], "output_dim": 10}}

    cfg example for Python file override:
        {"Python": {"pyfile": "./models/custom_model.py",
                    "model_name": "MyCustomModel",
                    "Model_Params": {...}}}

    registry: a Registry class with .get() and .all() methods
    """
    key = next(iter(cfg))
    params = cfg[key]

    # Look up in registry
    cls = registry.get(key)

    return cls(**params)