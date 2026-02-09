import configparser
from .base_loader import BaseLoader

class INILoader(BaseLoader):
    def load(self, path: str):
        config = configparser.ConfigParser()
        config.read(path)
        # Flatten sections or just return as dict
        data = {}
        for section in config.sections():
            for key, value in config.items(section):
                data[key.upper()] = value
        # Also check for default section
        for key, value in config.defaults().items():
            data[key.upper()] = value
        return data
