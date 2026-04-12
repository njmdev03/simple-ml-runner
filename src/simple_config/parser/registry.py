from typing import Union, Type, Any, List
from pathlib import Path

from simple_registries import Registry, AbstractClassRegistry

from simple_config.parser.ini import INIParser
from simple_config.parser.json import JSONParser
from simple_config.parser.yaml import YAMLParser
from simple_config.parser.toml import TOMLParser


class ParserRegistry(AbstractClassRegistry):
    """Global Registry of configuration parsers indexed by file extension
    (case in-sensitive).
    """
    _registry = Registry()


    @classmethod
    def register(cls, parser_cls: Type, *extensions: str):
        """Method to register a parser for given file extensions.

        Args:
            *extensions: One or more file extensions (e.g., 'yaml', 'json').

        Returns:
            The decorator function.
        """
        map(cls._normalize_extension, extensions)

        cls._registry.register(parser_cls, *extensions)

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
    def remove(cls, key: str) -> Union[Any, None]:
        return cls._registry.remove(key)

    @classmethod
    def deregister(cls, item: Any) -> Union[List[Any], None]:
        return cls._registry.deregister(item)

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
    def register_builtin_providers(cls):
        """Trigger the registration of basic parsers bundled in the library. This allows users to provide their own
        overrides for the default providers if these extensions are loaded later than user code.
        """
        cls.register(INIParser, "ini", "cfg")
        cls.register(JSONParser, "json")
        cls.register(YAMLParser, "yaml", "yml")
        cls.register(TOMLParser, "toml")


def ConfigParser(*extensions: str):
    """Decorator to register a config parser for one or more file extensions."""
    return ParserRegistry.decorator(*extensions)
