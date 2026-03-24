from ml_runner.core.registries.base import BaseRegistry


class TaskRegistry(BaseRegistry):
    _registry = {}


def Task(*names: str):
    return TaskRegistry.register(*names)
