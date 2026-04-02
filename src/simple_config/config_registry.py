from typing import Type, Dict, List, Callable
from dataclasses import dataclass, fields, is_dataclass
from simple_config.exceptions import ConfigNotFoundError, ConfigValidationError

class ConfigRegistry:
    """Registry of named config dataclasses."""
    _registry: Dict[str, Type] = {}

    @classmethod
    def register(cls, name: str, cfg_cls: Type):
        if not is_dataclass(cfg_cls):
            raise ConfigValidationError(f"Config for {name} must be a dataclass")
        cls._registry[name] = cfg_cls

    @classmethod
    def get(cls, name: str) -> Type:
        if name not in cls._registry:
            raise ConfigNotFoundError(f"Config schema '{name}' not found in registry. Registered: {cls.all()}")
        return cls._registry[name]

    @classmethod
    def all(cls):
        return list(cls._registry.keys())

def Config(name: str) -> Callable[[Type], Type]:
    """Decorator to register a config dataclass."""
    def decorator(cfg_cls: Type) -> Type:
        ConfigRegistry.register(name, cfg_cls)
        return cfg_cls
    return decorator
