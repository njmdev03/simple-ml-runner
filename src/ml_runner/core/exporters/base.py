from typing import Any

from ml_runner.core.config.schema import RunConfig
from ml_runner.core.cli_interface.cli_registry import CLIRegistry


class BaseExporter:
    """Base class for post-run exporters.

    Subclasses should implement `init()` for any early setup, `register_cli_arguments(parser)`
    to add CLI flags, and `run(exporter_config, global_config)` to execute the export.
    """
    def init(self):
        return None

    def register_cli_arguments(self, cli_registry: CLIRegistry):
        return None

    def run(self, exporter_config: Any, global_config: RunConfig):
        raise NotImplementedError()
