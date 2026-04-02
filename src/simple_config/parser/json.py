import json
from simple_config.parser.registry import ConfigParser
from simple_config.parser.base import ConfigParser as BaseParser

@ConfigParser("json")
class JSONParser(BaseParser):
    def load(self, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
