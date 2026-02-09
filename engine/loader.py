import importlib.util
import sys
from pathlib import Path

# Used for loading Models or datasets from .py file paths.
def load_from_pyscript(script_path, attribute_name):
    path = Path(script_path).resolve()
    module_name = path.stem

    original_sys_path = list(sys.path)
    sys.path.insert(0, str(path.parent))

    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module from {script_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # Support trying multiple attributes
        attrs = [attribute_name] if isinstance(attribute_name, str) else attribute_name
        for attr in attrs:
            if hasattr(module, attr):
                return getattr(module, attr)

        raise AttributeError(f"Module {module_name} has none of the attributes: {attrs}")
    finally:
        sys.path = original_sys_path
