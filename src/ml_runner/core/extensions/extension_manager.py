from typing import Dict, Type
from pathlib import Path
import logging

from ml_runner.core.registries.extension import ExtensionRegistry, ExtensionConfigRegistry
from ml_runner.core.extensions.base_extension import BaseExtension
from ml_runner.core.config.schema import RunConfig
from ml_runner.core.engine.event_manager import EventManager
from ml_runner.core.registries.callbacks import attach
from ml_runner.core.extensions.toml_loader import load_toml_plugins


class ExtensionManager:
    """
    Detects, constructs, and manages all extensions.
    Extensions are stateless until callbacks are created, so instances
    can be reused across multiple jobs.
    """

    def __init__(self):
        self._extensions: Dict[str, BaseExtension] = {}

    """
    Finds extensions, allowing them to register themselves, models
    datasets, and similar handlers.

    Constructs extensions, calling their register() method
    """
    def register_extensions(self):
        # Flush current extensions
        self._extensions = {}

        # Find extensions at known locations and load their python modules
        # Triggers registration via decorators.
        self._discover_extensions()

        # Construct extensions that registered to the ExtensionRegistry
        # Calls init() so that extensions can manually register features
        self._construct_extensions()

    # -------------------------
    # Discover all available extensions
    # -------------------------
    def _discover_extensions(self):
        """
        Load extensions into python path so that decorators trigger registrations
        """
        # Add built-in extensions
        import ml_runner.extensions

        # # Scan extension folders for ml-extension.toml files
        # # Project-local ./extensions/ then user-local ~/.config/ml_runner/extensions/
        # candidate_dirs = []
        # try:
        #     cwd = Path.cwd()
        #     candidate_dirs.append(cwd / "extensions")
        # except Exception:
        #     pass

        # try:
        #     home = Path.home()
        #     candidate_dirs.append(home / ".config" / "ml_runner" / "extensions")
        # except Exception:
        #     pass

        # logger = logging.getLogger(__name__)

        # for d in candidate_dirs:
        #     if not d.exists() or not d.is_dir():
        #         continue

        #     # # Only import top-level .py files found directly under the
        #     # # candidate directory. Package-based imports (dirs with
        #     # # __init__.py) are intentionally NOT supported to keep discovery
        #     # # simple and predictable.
        #     # for child in sorted(d.iterdir()):
        #     #     try:
        #     #         if child.is_file() and child.suffix == ".py":
        #     #             mod_name = f"ml_runner_extension_{child.stem}"
        #     #             spec = importlib.util.spec_from_file_location(mod_name, str(child))
        #     #             if spec and spec.loader:
        #     #                 mod = importlib.util.module_from_spec(spec)
        #     #                 try:
        #     #                     spec.loader.exec_module(mod)
        #     #                     logger.info("Imported extension module from file: %s", str(child))
        #     #                 except Exception as ie:
        #     #                     logger.warning("Failed to import extension file '%s': %s", child, ie)
        #     #     except Exception as e:
        #     #         logger.warning("Error while discovering extensions in '%s': %s", str(d), e)

        #     # Also allow one-deep subfolders that include a standard TOML
        #     # entrypoint file which points to a python file or module to import.
        #     try:
        #         load_toml_plugins(d, filename="ml_runner.toml")
        #     except Exception:
        #         logger.debug("No TOML plugins loaded from %s", d)

    # -------------------------
    # Construct instances from discovered classes
    # -------------------------
    def _construct_extensions(self):
        """
        Construct extension instances from discovered classes.
        """
        self._extensions.clear()

        for name in ExtensionRegistry.all():
            ext = ExtensionRegistry.get(name)()
            ext.init()
            self._extensions[name] = ext

    # -------------------------
    # Config class helper
    # -------------------------
    def get_config_classes(self) -> Dict[str, Type]:
        """
        Returns a mapping of registered extension names to their expected Config dataclass
        using ExtensionConfigRegistry.
        """
        classes = {}
        # Use the ExtensionConfigRegistry directly so callers don't need to
        # construct or initialize extensions just to discover their config classes.
        for name in ExtensionConfigRegistry.all():
            try:
                cfg_cls = ExtensionConfigRegistry.get(name)
            except ValueError:
                cfg_cls = None
            if cfg_cls:
                classes[name] = cfg_cls
        return classes

    # -------------------------
    # Callbacks
    # -------------------------
    def before_job(self, config: RunConfig, event_manager: EventManager):
        """
        Calls each extension to setup itself, passing the correctly
        parsed sub-config for that extension.
        """
        for name, ext in self._extensions.items():
            ext_cfg = config.extensions.get(name, None)
            # Attach any decorator-registered methods on the extension instance
            attach(ext, event_manager)

            ext.setup(config=ext_cfg, global_config=config, event_manager=event_manager)

    # -------------------------
    # CLI Arguments
    # -------------------------
    def register_cli_arguments(self, parser):
        """
        Calls each extension to register its CLI arguments.
        """
        for ext in self._extensions.values():
            ext.register_cli_arguments(parser)

    # -------------------------
    # Utility
    # -------------------------
    def get_extensions(self):
        return list(self._extensions.values())
