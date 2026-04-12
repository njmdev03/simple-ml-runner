import yaml
from pathlib import Path

from simple_config.parser.base import BaseParser


class YAMLParser(BaseParser):
    """Parser for YAML configuration files."""

    @classmethod
    def load(cls, path: Path) -> dict:
        """Load a YAML file.

        Args:
            path: The path to the YAML file.

        Returns:
            The parsed data.
        """
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
