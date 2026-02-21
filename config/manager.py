import os
import argparse
from typing import Dict, Any, List
from pathlib import Path

from .json_loader import JSONLoader
from .yaml_loader import YAMLLoader
from .toml_loader import TOMLLoader
from .ini_loader import INILoader
from .py_loader import PYLoader
from .config import Config, ResolvedConfig

class ConfigManager:
    def __init__(self):
        self.loaders = {
            '.json': JSONLoader(),
            '.yaml': YAMLLoader(),
            '.yml': YAMLLoader(),
            '.toml': TOMLLoader(),
            '.ini': INILoader(),
            '.py': PYLoader()
        }

    def load_config_tree(self, initial_paths: List[str]) -> Config:
        print("Loading configs...")

        # loaded_configs = []
        # stack = [(Path(p).resolve(), 0) for p in reversed(initial_paths)]
        visited = set()

        # We need to load in a way that respects the bottom-up priority
        # Let's do a post-order traversal of the config tree.
        # We'll store (content_dict, base_path) so we can resolve relative paths
        tree_order = []

        def traverse(path: Path):
            resolved_path = path.resolve()
            if resolved_path in visited:
                return
            visited.add(resolved_path)

            print(f"Processing config {resolved_path}")

            ext = resolved_path.suffix.lower()
            loader = self.loaders.get(ext)
            if not loader:
                print(f"No valid loader found for {resolved_path}")
                return

            print(f"Using loader {loader}")

            content = loader.load(str(resolved_path))

            # Check for nested configs
            nested = content.get('CONFIG', [])
            if isinstance(nested, str):
                nested = [nested]
                print(f"Found dependent configs {nested}")

            for n_path in nested:
                n_full_path = resolved_path.parent / n_path
                traverse(n_full_path)

            print(f"Adding content of {resolved_path} to tree")

            tree_order.append((content, resolved_path.parent))

        for p in initial_paths:
            traverse(Path(p))

        # Merge tree_order (Bottom-up) into typed Config using the new
        # `Config.from_dict` + `merge` semantics so that per-file provenance
        # and DefaultValue handling are respected.
        final_config = Config()
        for cfg, base in tree_order:
            inst = Config.from_dict(cfg, base_path=base)
            final_config.merge(inst)

        # Do not call `finalize()` here; resolved defaults are applied when
        # consumers call `Config.resolve()` (which performs fallbacks for
        # testing values). Return the merged `Config` as-is.
        return final_config

    def apply_env_overrides(self, config: Config):
        overrides = {}
        # iterate known config fields
        for key in config.to_dict().keys():
            if key in os.environ:
                val = os.environ[key]
                overrides[key] = val

        if overrides:
            config.update_from_dict(overrides)
        return config

    def apply_cli_overrides(self, config: Config, args: argparse.Namespace):
        arg_dict = vars(args)

        # Training logic (inverted flags)
        if arg_dict.get('dont_train'):
            config.update_from_dict({'TRAIN': False})
        elif arg_dict.get('train'):
            config.update_from_dict({'TRAIN': True})

        # Testing logic (inverted flags)
        if arg_dict.get('dont_test'):
            config.update_from_dict({'TEST': False})
        elif arg_dict.get('test'):
            config.update_from_dict({'TEST': True})

        # Generic overrides for other keys
        # We only want to override if the user EXPLICITLY provided the argument.
        # For store_true flags, v will be False by default.
        # We only override if v is True OR if v is not a boolean (like a string path).

        # Keys that use action='store_true'
        bool_flags = [
            'test_while_training', 'test_on_training_data',
            'test_checkpoints', 'silent', 'profile', 'show'
        ]

        overrides = {}
        for k, v in arg_dict.items():
            if v is None:
                continue
            if k in ['train', 'dont_train', 'test', 'dont_test', 'dont_save_metadata', 'config']:
                continue

            # If it's a known boolean flag and it's False, it means it wasn't passed.
            if k in bool_flags and v is False:
                continue

            upper_k = k.upper().replace("-", "_")
            overrides[upper_k] = v

        if overrides:
            config.update_from_dict(overrides)

        return config
