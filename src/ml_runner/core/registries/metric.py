from ml_runner.core.registries.base import BaseRegistry


class MetricRegistry(BaseRegistry):
    _registry = {}


def Metric(*names: str):
    return MetricRegistry.register(*names)
