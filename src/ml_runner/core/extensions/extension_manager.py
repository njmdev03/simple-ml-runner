from typing import Dict, Any, Type

from ml_runner.core.registries import ExtensionRegistry
from ml_runner.core.extensions.base_extension import BaseExtension
from ml_runner.core.config.schema import RunConfig

class ExtensionManager:
    """
    Detects, constructs, and manages all extensions.
    """

    def __init__(self):
        self._extensions: Dict[str, BaseExtension] = {}

    def get_config_classes(self) -> Dict[str, Type]:
        """
        Returns a mapping of registered extension names to their expected Config dataclass.
        """
        from ml_runner.core.registries import ExtensionRegistry
        classes = {}
        for name in ExtensionRegistry.all():
            ext_cls = ExtensionRegistry.get(name)
            cfg_cls = getattr(ext_cls, "_config_class", None)
            if cfg_cls:
                classes[name] = cfg_cls
        return classes

    def discover_and_construct(self, global_config: RunConfig = None):
        """
        Construct all registered extensions.
        """
        from ml_runner.core.registries import ExtensionRegistry

        for name in ExtensionRegistry.all():
            ext_cls: Type[BaseExtension] = ExtensionRegistry.get(name)
            ext = ext_cls()

            if hasattr(ext, 'init'):
                ext.init()             # optional registration hooks
            self._extensions[name] = ext

    def create_callbacks(self, config: RunConfig) -> list:
        """
        Calls each extension to create callbacks, passing the correctly
        parsed sub-config for that extension.
        """
        all_callbacks = []

        for name, ext in self._extensions.items():
            ext_cfg = config.extensions.get(name) if config is not None else None

            # Call extension's create_callbacks with pre-parsed config injection
            callbacks = ext.create_callbacks(global_config=config, config=ext_cfg)
            if callbacks:
                all_callbacks.extend(callbacks)

        return all_callbacks

    def all_extensions(self):
        return list(self._extensions.values())