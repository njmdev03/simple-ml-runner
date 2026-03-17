from .base import BaseRegistry


class ConfigRegistry(BaseRegistry):
    _registry = {}


def Config(*names: str):
    return ConfigRegistry.register(*names)
