from typing import Type

from .base import BaseRegistry


class ExtensionRegistry(BaseRegistry):
    _registry = {}


class ExtensionConfigRegistry(BaseRegistry):
    _registry = {}


def Extension(name: str, config_class: Type = None):
    """
    Registers an extension class and optionally its associated config class.
    """
    def decorator(cls):
        cls._extension_name = name
        cls._config_class = config_class

        # Register the extension class
        ExtensionRegistry.register(name)(cls)

        # Register the config class (if provided)
        if config_class is not None:
            ExtensionConfigRegistry.register(name)(config_class)

        return cls
    return decorator
