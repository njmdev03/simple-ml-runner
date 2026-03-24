from typing import Type

from ml_runner.core.registries.base import BaseRegistry


def resolve_component(cfg: dict, registry: Type[BaseRegistry]):
    """
    Resolve and instantiate a component from a registry.

    Example:
        {"mlp": {"input_dim": 784}}

    Returns:
        instance of registered class
    """
    if not cfg:
        raise ValueError("resolve_component: empty config")

    if len(cfg) != 1:
        raise ValueError(
            f"resolve_component expects a single key, got: {list(cfg.keys())}"
        )

    name, params = next(iter(cfg.items()))
    params = params or {}

    cls = registry.get(name)

    try:
        return cls(**params)
    except TypeError as e:
        raise TypeError(
            f"Error constructing '{name}' with params {params}: {e}"
        ) from e
