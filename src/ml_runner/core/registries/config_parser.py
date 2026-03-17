from .base import BaseRegistry


class ConfigParserRegistry(BaseRegistry):
    _registry = {}


def ConfigParser(*names: str):
    return ConfigParserRegistry.register(*names)
