import importlib.util
import sys
import os
from pathlib import Path


# Used for loading Models or datasets from .py file paths.
def load_from_pyscript(script_path, attribute_name):
    path = Path(script_path).resolve()
    module_name = path.stem

    original_sys_path = list(sys.path)
    original_cwd = os.getcwd()
    sys.path.insert(0, str(path.parent))

    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module from {script_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module

        # Ensure relative file operations inside the script resolve from the
        # script's directory by temporarily changing cwd.
        os.chdir(str(path.parent))

        spec.loader.exec_module(module)

        # Support trying multiple attributes
        attrs = [attribute_name] if isinstance(attribute_name, str) else attribute_name
        for attr in attrs:
            if hasattr(module, attr):
                return getattr(module, attr)

        raise AttributeError(f"Module {module_name} has none of the attributes: {attrs}")
    finally:
        # restore process state
        os.chdir(original_cwd)
        sys.path = original_sys_path
