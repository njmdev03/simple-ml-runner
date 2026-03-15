from typing import Callable, Dict


class BaseRegistry:
    """
    Generic base registry for storing components keyed by name.
    Subclasses can specialize for models, datasets, optimizers, etc.
    """
    _registry: Dict[str, Callable] = {}

    @classmethod
    def register(cls, *names):
        """
        Decorator to register a class or function under one or more names.
        """
        def decorator(obj):
            for name in names:
                cls._registry[name.lower()] = obj
            return obj
        return decorator

    @classmethod
    def get(cls, name: str):
        """
        Fetch a registered object by name.
        """
        name = name.lower()
        if name not in cls._registry:
            raise ValueError(f"No component registered under name: '{name}'")
        return cls._registry[name]

    @classmethod
    def all(cls):
        """
        Return all registered names.
        """
        return list(cls._registry.keys())


class ConfigRegistry(BaseRegistry):
    _registry = {}


class ModelRegistry(BaseRegistry):
    _registry = {}


class DatasetRegistry(BaseRegistry):
    _registry = {}


class OptimizerRegistry(BaseRegistry):
    _registry = {}


class LossRegistry(BaseRegistry):
    _registry = {}


class MetricRegistry(BaseRegistry):
    _registry = {}


class SchedulerRegistry(BaseRegistry):
    _registry = {}


def resolve_component(cfg: dict, registry: BaseRegistry) -> object:
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
