import yaml
from simple_config.parser.registry import ConfigParser
from simple_config.parser.base import ConfigParser as BaseParser

@ConfigParser("yaml", "yml")
class YAMLParser(BaseParser):
    def load(self, path: str) -> dict:
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
