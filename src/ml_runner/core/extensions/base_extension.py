from typing import Any, Optional

from ml_runner.core.config.schema import RunConfig


class BaseExtension:
    """
    Base class for all extensions.
    Can optionally define config schemas and/or callbacks.
    """
    def create_callbacks(self, global_config: RunConfig, config: Optional[Any] = None):
        return []
