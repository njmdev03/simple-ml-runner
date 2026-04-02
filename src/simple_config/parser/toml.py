try:
    import tomllib  # Python 3.11+
except ImportError:
    import toml as tomllib  # type: ignore

from simple_config.parser.registry import ConfigParser
from simple_config.parser.base import BaseParser

@ConfigParser("toml")
class TOMLParser(BaseParser):
    """Parser for TOML configuration files."""

    def load(self, path: str) -> dict:
        """Load a TOML file.

        Args:
            path: The path to the TOML file.

        Returns:
            The parsed data.
        """
        with open(path, "rb") as f:
            return tomllib.load(f) or {}
