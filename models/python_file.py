# models/python_model.py
import importlib.util
import sys
from pathlib import Path
from registries.model_registry import ModelRegistry

@ModelRegistry.register("Python")
class PythonModel:
    """
    Wrapper for dynamically loading a model from a Python file.
    """

    def __init__(self, pyfile: str, model_name: str, Model_Params: dict = None):
        """
        pyfile: path to Python file containing the model class
        model_name: name of the class in that file
        Model_Params: dict of parameters to pass to the class
        """
        self.pyfile = Path(pyfile).resolve()
        self.model_name = model_name
        self.params = Model_Params or {}

        self.model = self._load_model()

    def _load_model(self):
        # Load the module dynamically
        spec = importlib.util.spec_from_file_location(self.model_name, self.pyfile)
        module = importlib.util.module_from_spec(spec)
        sys.modules[self.model_name] = module
        spec.loader.exec_module(module)

        # Get the class and instantiate it
        cls = getattr(module, self.model_name)
        return cls(**self.params)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def __getattr__(self, item):
        # Delegate attribute access to the underlying model
        return getattr(self.model, item)