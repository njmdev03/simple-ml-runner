import configparser
from simple_config.parser.registry import ConfigParser
from simple_config.parser.base import BaseParser

@ConfigParser("ini", "cfg")
class INIParser(BaseParser):
    """Parser for INI configuration files."""

    def load(self, path: str) -> dict:
        """Load an INI file.

        Args:
            path: The path to the INI file.

        Returns:
            The parsed data as a dictionary of section -> items.
        """
        config = configparser.ConfigParser()
        config.read(path, encoding="utf-8")
        return {section: dict(config.items(section)) for section in config.sections()}
