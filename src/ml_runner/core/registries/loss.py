from ml_runner.core.registries.base import BaseRegistry


class LossRegistry(BaseRegistry):
    _registry = {}


def Loss(*names: str):
    return LossRegistry.register(*names)
