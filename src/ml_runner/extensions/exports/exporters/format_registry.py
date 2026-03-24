from ml_runner.core.registries.base import BaseRegistry


class DataFormatRegistry(BaseRegistry):
    _registry = {}


def DataFormat(*names: str):
    """Decorator to register a data format handler local to the exporter extension.

    Usage:
        @DataFormat('csv')
        def write_csv(rows, cols, out_path):
            ...
    """
    return DataFormatRegistry.register(*names)
