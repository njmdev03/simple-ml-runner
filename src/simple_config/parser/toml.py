try:
    import tomllib  # Python 3.11+
except ImportError:
    import toml as tomllib  # type: ignore

from simple_config.parser.registry import ConfigParser
from simple_config.parser.base import ConfigParser as BaseParser

@ConfigParser("toml")
class TOMLParser(BaseParser):
    def load(self, path: str) -> dict:
        with open(path, "rb") as f:
            return tomllib.load(f)
