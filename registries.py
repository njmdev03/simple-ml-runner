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
