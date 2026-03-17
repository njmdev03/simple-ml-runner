from .base import BaseRegistry


class ExtensionRegistry(BaseRegistry):
    _registry = {}


def Extension(*names: str):
    return ExtensionRegistry.register(*names)
