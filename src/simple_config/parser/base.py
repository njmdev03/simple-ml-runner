from abc import ABC, abstractmethod
from pathlib import Path

class BaseParser(ABC):
    """Abstract base class for all configuration parsers."""

    @classmethod
    @abstractmethod
    def load(path: Path) -> dict:
        """Load a configuration file into a dictionary.

        Args:
            path: The path to the configuration file.

        Returns:
            The parsed data as a dictionary.
        """
        pass
