import yaml
from pathlib import Path

from simple_config.parser.registry import ConfigParser
from simple_config.parser.base import BaseParser

@ConfigParser("yaml", "yml")
class YAMLParser(BaseParser):
    """Parser for YAML configuration files."""

    def load(self, path: Path) -> dict:
        """Load a YAML file.

        Args:
            path: The path to the YAML file.

        Returns:
            The parsed data.
        """
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
