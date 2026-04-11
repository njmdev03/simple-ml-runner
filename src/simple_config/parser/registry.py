from typing import Union
from pathlib import Path

from simple_registries import Registry, AbstractClassRegistry

from simple_config.parser.base import BaseParser


class ParserRegistry(AbstractClassRegistry):
    """Global Registry of configuration parsers indexed by file extension
    (case in-sensitive).
    """
    _registry = Registry()


    @staticmethod
    def _normalize_extension(extension: str) -> str:
        """Normalize the passed extension, stripping the  leading "." if it is present.

        Args:
            extension (str): The extension string to normalize

        Returns:
            str: The extension in a normalized format
        """
        return extension.lstrip(".").lower()

    @classmethod
    def register(cls, parser_cls: BaseParser, *extensions: str):
        """Method to register a parser for given file extensions.

        Args:
            *extensions: One or more file extensions (e.g., 'yaml', 'json').

        Returns:
            The decorator function.
        """
        extensions.map(cls._normalize_extension)

        cls._registry.register(parser_cls, extensions)

    @classmethod
    def get(cls, extension: str):
        """Get a parser by its extension.

        Args:
            extension: The file extension.

        Returns:
            The parser instance or None.
        """
        # Normalize extension: remove leading dot if present
        extension = cls._normalize_extension(extension)
        return cls._registry.get(extension)

    @classmethod
    def clear(cls):
        cls._registry.clear()

    @classmethod
    def all(cls):
        return cls._registry.keys()

    @classmethod
    def decorator(cls, *extensions) -> callable:
        def decorator(cls):
            ParserRegistry.register(cls, *extensions)

            return cls

        return decorator

    @classmethod
    def get_parser(cls, path: Union[Path, str]):
        """Get the appropriate parser instance for a given file path.

        Args:
            path: The path to the file.

        Returns:
            A parser instance.

        Raises:
            ValueError: If no parser is registered for the file extension.
        """
        if path is str:
            path = Path(path)

        extension = cls._normalize_extension(path.suffix)

        return cls._registry.get(extension)


def ConfigParser(cls, *extensions: str):
    """Decorator to register a config parser for one or more file extensions."""
    return ParserRegistry.decorator(extensions)
