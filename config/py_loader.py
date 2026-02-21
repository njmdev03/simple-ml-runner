import importlib.util
import sys
from pathlib import Path
from .base_loader import BaseLoader
import os

class PYLoader(BaseLoader):
    def load(self, path: str):
        file_path = Path(path).resolve()
        module_name = file_path.stem

        # Add the directory to sys.path so the module can import from its own directory
        original_sys_path = sys.path.copy()
        sys.path.insert(0, str(file_path.parent))

        original_cwd = os.getcwd()
        os.chdir(str(file_path.parent))

        try:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load module from {path}")

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # Return all uppercase variables
            return {k: v for k, v in vars(module).items() if k.isupper()}
        finally:
            os.chdir(original_cwd)
            sys.path = original_sys_path
