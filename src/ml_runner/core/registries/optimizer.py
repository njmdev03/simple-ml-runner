from .base import BaseRegistry


class OptimizerRegistry(BaseRegistry):
    _registry = {}


def Optimizer(*names: str):
    return OptimizerRegistry.register(*names)
