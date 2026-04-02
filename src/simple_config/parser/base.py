from abc import ABC, abstractmethod

class BaseParser(ABC):
    """Abstract base class for all configuration parsers."""

    @abstractmethod
    def load(self, path: str) -> dict:
        """Load a configuration file into a dictionary.

        Args:
            path: The path to the configuration file.

        Returns:
            The parsed data as a dictionary.
        """
        pass
