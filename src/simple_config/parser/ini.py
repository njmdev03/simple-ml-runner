import configparser
from simple_config.parser.registry import ConfigParser
from simple_config.parser.base import ConfigParser as BaseParser

@ConfigParser("ini", "cfg")
class INIParser(BaseParser):
    def load(self, path: str) -> dict:
        config = configparser.ConfigParser()
        config.read(path, encoding="utf-8")
        return {section: dict(config.items(section)) for section in config.sections()}
