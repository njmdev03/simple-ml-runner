import toml
from .base_loader import BaseLoader

class TOMLLoader(BaseLoader):
    def load(self, path: str):
        with open(path, 'r') as f:
            return toml.load(f)
