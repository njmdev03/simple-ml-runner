import configparser
from pathlib import Path

from simple_config.parser.base import BaseParser


class INIParser(BaseParser):
    """Parser for INI configuration files."""

    @classmethod
    def load(cls, path: Path) -> dict:
        """Load an INI file.

        Args:
            path: The path to the INI file.

        Returns:
            The parsed data as a dictionary of section -> items.
        """
        config = configparser.ConfigParser()
        config.read(path, encoding="utf-8")
        return {section: dict(config.items(section)) for section in config.sections()}
