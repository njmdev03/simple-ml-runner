import json
from pathlib import Path

from simple_config.parser.base import BaseParser


class JSONParser(BaseParser):
    """Parser for JSON configuration files."""

    @classmethod
    def load(cls, path: Path) -> dict:
        """Load a JSON file.

        Args:
            path: The path to the JSON file.

        Returns:
            The parsed data.
        """
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
