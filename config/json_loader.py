import json
from .base_loader import BaseLoader

class JSONLoader(BaseLoader):
    def load(self, path: str):
        with open(path, 'r') as f:
            return json.load(f)
