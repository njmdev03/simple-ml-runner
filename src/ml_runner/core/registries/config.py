from .base import BaseRegistry


class ConfigRegistry(BaseRegistry):
    _registry = {}


class NamedConfigRegistry(BaseRegistry):
    _registry = {}


def Config(*names: str):
    return ConfigRegistry.register(*names)


def NamedConfig(*names: str):
    return NamedConfigRegistry.register(*names)
