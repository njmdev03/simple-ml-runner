import json
from simple_config.parser.registry import ConfigParser
from simple_config.parser.base import BaseParser

@ConfigParser("json")
class JSONParser(BaseParser):
    """Parser for JSON configuration files."""

    def load(self, path: str) -> dict:
        """Load a JSON file.

        Args:
            path: The path to the JSON file.

        Returns:
            The parsed data.
        """
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
