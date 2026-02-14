import os
import argparse
from typing import Dict, Any, List
from pathlib import Path

from .json_loader import JSONLoader
from .yaml_loader import YAMLLoader
from .toml_loader import TOMLLoader
from .ini_loader import INILoader
from .py_loader import PYLoader

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

        self.defaults = {
            # Meta
            'SILENT': False,
            'PROFILE': False,
            # 'PROFILE_OUTPUT': None, # Default to None, handled in code

            # Device
            'DEVICES': ['cpu'],

            # Data
            # 'TRAIN_DATASET': None,
            # 'TEST_DATASET': None,
            'FINAL_OUTPUT_PATH': 'model_final.pt',

            # Training Flags
            'TRAIN': True,
            'TEST': True, # Default to True, but typically controlled by mode

            # Training Hyperparameters
            'BATCH_SIZE': 32,
            'LEARNING_RATE': 0.001,
            'EPOCHS': 1,
            'OPTIMIZER': 'Adam',
            'TRAIN_CRITERION': 'CrossEntropyLoss',

            # Checkpointing
            'CHECK_RATE': 0,
            'CHECK_MODEL_DIR': 'checkpoints/',
            'CHECK_MODEL_NAME': 'model_epoch_$epoch',
            'SAVE_METADATA': True,
            'RESUME': False,

            # Early Halt
            'EARLY_HALT_CONDITION': 'None',
            'EARLY_HALT_THRESHOLD': 0.0,

            # Testing & Evaluation
            # 'TESTING_BATCH_SIZE': 32, # Will likely be overriden by BATCH_SIZE if not present, but good safely
            # 'TESTING_CRITERION': [],
            'TEST_ON_TRAINING_DATA': False,
            'TEST_WHILE_TRAINING': False,
            'TEST_CHECKPOINTS': False,
            # 'SAVE_TESTS': None,

            # Visualization
            'VIS_TYPE': ['all'],
            'VIS_OUTPUT_DIR': 'vis',
            'VIS_FORMAT': 'png',
            'VIS_LAYOUT': 'individual',
            'SHOW': False,
            'NUM_SAMPLES': 0,
            'VIS_DATASETS': ['testing'],
            'VIS_METRICS': ['all']
        }

    def load_config_tree(self, initial_paths: List[str]) -> Dict[str, Any]:
        print("Loading configs...")

        # loaded_configs = []
        # stack = [(Path(p).resolve(), 0) for p in reversed(initial_paths)]
        visited = set()

        # We need to load in a way that respects the bottom-up priority
        # Let's do a post-order traversal of the config tree.
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

            tree_order.append(content)

        for p in initial_paths:
            traverse(Path(p))

        # Merge tree_order (Bottom-up)
        # print(f"\n Processing config tree {tree_order}")
        final_config = self.defaults.copy()
        for cfg in tree_order:
            final_config.update(cfg)

        return final_config

    def apply_env_overrides(self, config: Dict[str, Any]):
        for key in config.keys():
            if key in os.environ:
                val = os.environ[key]
                # Simple type inference for env vars
                if val.lower() == 'true': val = True
                elif val.lower() == 'false': val = False
                elif val.isdigit(): val = int(val)
                else:
                    try: val = float(val)
                    except ValueError: pass
                config[key] = val
        return config

    def apply_cli_overrides(self, config: Dict[str, Any], args: argparse.Namespace):
        arg_dict = vars(args)

        # Training logic (inverted flags)
        if arg_dict.get('dont_train'):
            config['TRAIN'] = False
        elif arg_dict.get('train'):
            config['TRAIN'] = True

        # Testing logic (inverted flags)
        if arg_dict.get('dont_test'):
            config['TEST'] = False
        elif arg_dict.get('test'):
            config['TEST'] = True

        # Generic overrides for other keys
        # We only want to override if the user EXPLICITLY provided the argument.
        # For store_true flags, v will be False by default.
        # We only override if v is True OR if v is not a boolean (like a string path).

        # Keys that use action='store_true'
        bool_flags = [
            'test_while_training', 'test_on_training_data',
            'test_checkpoints', 'silent', 'profile', 'show'
        ]

        for k, v in arg_dict.items():
            if v is None:
                continue
            if k in ['train', 'dont_train', 'test', 'dont_test', 'dont_save_metadata', 'config']:
                continue

            # If it's a known boolean flag and it's False, it means it wasn't passed.
            if k in bool_flags and v is False:
                continue

            upper_k = k.upper().replace("-", "_")
            config[upper_k] = v

        return config
