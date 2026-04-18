from typing import Union, Type, Any, List
from pathlib import Path

from simple_registries import Registry

from simple_config.parser.ini import INIParser
from simple_config.parser.json import JSONParser
from simple_config.parser.yaml import YAMLParser
from simple_config.parser.toml import TOMLParser


class ParserRegistry(Registry):
    """Registry of configuration parsers indexed by file extension (case in-sensitive).
    """

    def register(self, parser_self: Type, *extensions: str):
        """Method to register a parser for given file extensions.

        Args:
            *extensions: One or more file extensions (e.g., 'yaml', 'json').

        Returns:
            The decorator function.
        """
        extensions = [self._normalize_extension(ext) for ext in extensions]

        super().register(parser_self, *extensions)

    def get(self, extension: str):
        """Get a parser by its extension.

        Args:
            extension: The file extension.

        Returns:
            The parser instance or None.
        """
        # Normalize extension: remove leading dot if present
        extension = self._normalize_extension(extension)
        return super().get(extension)

    def remove(self, key: str) -> Union[Any, None]:
        return super().remove(key)

    def deregister(self, item: Any) -> Union[List[Any], None]:
        return super().deregister(item)

    def clear(self):
        super().clear()

    def all(self):
        return super().all()

    def keys(self) -> List:
        """Get all of the registered keys.

        Returns:
            list: All of the registered keys.
        """
        return super().keys()

    # def decorator(self, *extensions) -> callable:
    #     def decorator(self):
    #         ParserRegistry.register(self, *extensions)

    #         return self

    #     return decorator

    def get_parser(self, path: Union[Path, str]):
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

        extension = self._normalize_extension(path.suffix)

        return super().get(extension)

    @staticmethod
    def _normalize_extension(extension: str) -> str:
        """Normalize the passed extension, stripping the  leading "." if it is present.

        Args:
            extension (str): The extension string to normalize

        Returns:
            str: The extension in a normalized format
        """
        return extension.lstrip(".").lower()

    def register_builtin_providers(self):
        """Trigger the registration of basic parsers bundled in the library. This allows users to provide their own
        overrides for the default providers if these extensions are loaded later than user code.
        """
        self.register(INIParser, "ini", "cfg")
        self.register(JSONParser, "json")
        self.register(YAMLParser, "yaml", "yml")
        self.register(TOMLParser, "toml")
