from .base import BaseRegistry


class DatasetRegistry(BaseRegistry):
    _registry = {}


def Dataset(*names: str):
    return DatasetRegistry.register(*names)
