from .base import BaseRegistry


class ModelRegistry(BaseRegistry):
    _registry = {}


def Model(*names: str):
    return ModelRegistry.register(*names)
