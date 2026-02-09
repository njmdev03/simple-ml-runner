import yaml
from .base_loader import BaseLoader

class YAMLLoader(BaseLoader):
    def load(self, path: str):
        with open(path, 'r') as f:
            return yaml.safe_load(f)
