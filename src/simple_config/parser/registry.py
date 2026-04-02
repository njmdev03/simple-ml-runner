from typing import Dict, Type, Callable, Optional, Any
from pathlib import Path

class ParserRegistry:
    _parsers: Dict[str, Any] = {}

    @classmethod
    def register(cls, *extensions: str):
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
        suffix = Path(path).suffix.lstrip(".")
        if not suffix:
            # Fallback for paths without extension or if suffix didn't work as expected
            suffix = path.split(".")[-1]
            
        if suffix not in cls._parsers:
            raise ValueError(f"No parser registered for extension: .{suffix} (from {path})")
        return cls._parsers[suffix]

    @classmethod
    def get(cls, extension: str):
        # Normalize extension: remove leading dot if present
        extension = extension.lstrip(".")
        return cls._parsers.get(extension)

def ConfigParser(*extensions: str):
    """Decorator to register a config parser for one or more file extensions."""
    return ParserRegistry.register(*extensions)
