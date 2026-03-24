from ml_runner.core.registries.base import BaseRegistry


class MetadataWriterRegistry(BaseRegistry):
    _registry = {}


def MetadataWriter(*names: str):
    return MetadataWriterRegistry.register(*names)
