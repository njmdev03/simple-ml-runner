from typing import Dict, Type, Any
from pathlib import Path

class ParserRegistry:
    """Registry of configuration parsers indexed by file extension."""
    _parsers: Dict[str, Any] = {}

    @classmethod
    def register(cls, *extensions: str):
        """Decorator to register a parser for given file extensions.

        Args:
            *extensions: One or more file extensions (e.g., 'yaml', 'json').

        Returns:
            The decorator function.
        """
        def decorator(parser_cls: Type):
            parser_instance = parser_cls()
            for ext in extensions:
                # Normalize extension: remove leading dot if present
                ext = ext.lstrip(".")
                cls._parsers[ext] = parser_instance
            return parser_cls
        return decorator

    @classmethod
    def get_parser(cls, path: str):
        """Get the appropriate parser instance for a given file path.

        Args:
            path: The path to the file.

        Returns:
            A parser instance.

        Raises:
            ValueError: If no parser is registered for the file extension.
        """
        suffix = Path(path).suffix.lstrip(".")
        if not suffix:
            # Fallback for paths without extension or if suffix didn't work as expected
            suffix = path.split(".")[-1]

        if suffix not in cls._parsers:
            raise ValueError(f"No parser registered for extension: .{suffix} (from {path})")
        return cls._parsers[suffix]

    @classmethod
    def get(cls, extension: str):
        """Get a parser by its extension.

        Args:
            extension: The file extension.

        Returns:
            The parser instance or None.
        """
        # Normalize extension: remove leading dot if present
        extension = extension.lstrip(".")
        return cls._parsers.get(extension)

def ConfigParser(*extensions: str):
    """Decorator to register a config parser for one or more file extensions."""
    return ParserRegistry.register(*extensions)
