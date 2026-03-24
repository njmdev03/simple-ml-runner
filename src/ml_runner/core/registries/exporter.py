from typing import Callable, Dict, Optional, Type

from ml_runner.core.registries.base import BaseRegistry


class ExporterRegistry(BaseRegistry):
    _registry: Dict[str, Callable] = {}

    @classmethod
    def all(cls):
        return list(cls._registry.keys())

class ExporterConfigRegistry(BaseRegistry):
    _registry: Dict[str, Type] = {}


def Exporter(name: str, config_class: Optional[Type] = None):
    """Decorator to register an exporter class and optionally its config dataclass.

    Usage:
        @Exporter('data', config_class=DataExportConfig)
        class DataExporter(BaseExporter):
            ...
    """
    def decorator(cls):
        # Register exporter class
        ExporterRegistry.register(name)(cls)

        # Register config class if provided
        if config_class is not None:
            ExporterConfigRegistry.register(name)(config_class)

        return cls

    return decorator
