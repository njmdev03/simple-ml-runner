from typing import Type, Dict, Callable, Optional
from dataclasses import dataclass, fields, is_dataclass
from simple_config.exceptions import ConfigNotFoundError, ConfigValidationError

class ConfigRegistry:
    """Registry of named config dataclasses and polymorphic mappings.

    This registry serves two purposes:
    1. Mapping schema names to root dataclasses.
    2. Mapping base dataclass types to a dictionary of "friendly name" -> "implementation dataclass".
    """
    _registry: Dict[str, Type] = {}
    _mappings: Dict[Type, Dict[str, Type]] = {}

    @classmethod
    def register(cls, name: str, cfg_cls: Type):
        """Register a dataclass with a schema name.

        Args:
            name: The schema name to register.
            cfg_cls: The dataclass to register.

        Raises:
            ConfigValidationError: If cfg_cls is not a dataclass.
        """
        if not is_dataclass(cfg_cls):
            raise ConfigValidationError(f"Config for {name} must be a dataclass")
        cls._registry[name] = cfg_cls

    @classmethod
    def register_mapping(cls, base_cls: Type, name: str, impl_cls: Type):
        """Register a polymorphic mapping for a base type.

        Args:
            base_cls: The base class or type that serves as the trigger.
            name: The friendly name used in the configuration (e.g., 'MLP').
            impl_cls: The actual dataclass implementation.
        """
        if base_cls not in cls._mappings:
            cls._mappings[base_cls] = {}
        cls._mappings[base_cls][name] = impl_cls

    @classmethod
    def get(cls, name: str) -> Type:
        """Get a registered dataclass by name.

        Args:
            name: The schema name.

        Returns:
            The registered dataclass type.

        Raises:
            ConfigNotFoundError: If the name is not registered.
        """
        if name not in cls._registry:
            raise ConfigNotFoundError(f"Config schema '{name}' not found in registry. Registered: {cls.all()}")
        return cls._registry[name]

    @classmethod
    def get_mapping(cls, base_cls: Type) -> Optional[Dict[str, Type]]:
        """Get the mapping for a base type, if any.

        Args:
            base_cls: The base type to look up.

        Returns:
            A dictionary mapping friendly names to dataclasses, or None.
        """
        return cls._mappings.get(base_cls)

    @classmethod
    def all(cls):
        """Return all registered schema names."""
        return list(cls._registry.keys())

def Config(name: str) -> Callable[[Type], Type]:
    """Decorator to register a config dataclass with a schema name.

    Args:
        name: The schema name.

    Returns:
        The decorator function.
    """
    def decorator(cfg_cls: Type) -> Type:
        ConfigRegistry.register(name, cfg_cls)
        return cfg_cls
    return decorator

def Mapping(base_cls: Type, name: str) -> Callable[[Type], Type]:
    """Decorator to register a polymorphic mapping for a base type.

    Args:
        base_cls: The base type trigger.
        name: The friendly name.

    Returns:
        The decorator function.
    """
    def decorator(cfg_cls: Type) -> Type:
        ConfigRegistry.register_mapping(base_cls, name, cfg_cls)
        return cfg_cls
    return decorator
