from typing import Any, Optional

from ml_runner.core.config.schema import RunConfig
from ml_runner.core.engine.event_manager import EventManager
from ml_runner.core.cli_interface.base_parser import BaseParser as Parser


class BaseExtension:
    """
    Base class for all extensions.
    Can optionally define config schemas and/or callbacks.
    """
    def init(self):
        pass

    def setup(self, event_manager: EventManager, global_config: RunConfig, config: Optional[Any] = None):
        """
        Setup the extension, register listeners, etc.
        """
        pass

    def register_cli_arguments(self, parser: Parser):
        """
        Allows extensions to register their own CLI arguments.
        """
        pass

    # def after_job(self):
    #     pass
