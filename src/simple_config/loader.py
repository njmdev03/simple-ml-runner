from pathlib import Path
from typing import Union, List, Optional, Set, Dict
from simple_config.parser.registry import ParserRegistry
from simple_config.utils import merge_dicts
from simple_config.exceptions import SimpleConfigError, ConfigNotFoundError


class ConfigLoader:
    _cache: Dict[Path, dict] = {}

    @classmethod
    def clear_cache(cls):
        _cache = {}

    @classmethod
    def _resolve_to_dict(cls, config: Union[Path],) -> dict:
        """Reads a file path to a dict. Will also read from a cache of file paths that have already been resolved.

        Args:
            config: string or pathlib path to a config file to be loaded to a dict. Files are resolved to dicts using
            the ParserRegistry or cached resolutions.

        Returns:
            dict: the loaded config dictionary
        """
        if cls._cache.get(config):
            return cls._cache.get(config)

        if not config.exists():
            raise ConfigNotFoundException

        if not ParserRegistry.get_parser(config):
            raise ConfigNotFoundException

        return ParserRegistry.get_parser(config).load(config)

    @classmethod
    def _load_config(
            cls,
            *config: Union[Dict, Path, str],
            seen: Optional[Set[Path]] = [],
            inheritance_key: str = "config"
        ) -> dict:
        # Resolve strings to a path
        if config is str:
            config = Path(config)

        # Resolve Paths to a dict
        if config is Path:
            top_dict = cls._resolve_to_dict(config)

            # Save explored paths for circular dependency checks.
            seen.append(config)
        else:
            top_dict = config

        inherits = top_dict.get(inheritance_key)

        if inherits:
            if inherits is str:
                inherits = [inherits]

            base_dict = {}

            for cfg in inherits:



        if config.get(inheritance_key):
            if config.get(inheritance_key) is list:
                for conf in config.get(inheritance_key):
                    pass


    def load_config(
        cls,
        *config: Union[Dict, Path, str],
        inheritance_key: str = "config"
        ) -> dict:
        return cls._load_config(*config, inheritance_key=inheritance_key)

def load_raw_config(
    path: Union[str, Path],
    seen: Optional[Set[Path]] = None,
    inheritance_key: str = "config",
) -> dict:
    """Load a configuration file and handle inheritance via the 'config' key.

    Detects circular dependencies and merges base configurations recursively.

    Args:
        path: The path to the configuration file.
        seen: A set of paths already seen in the inheritance chain to detect cycles.
        inheritance_key: The key used in the config file to specify base configs.
            Defaults to "config".

    Returns:
        A dictionary containing the merged configuration data.

    Raises:
        ConfigNotFoundError: If the configuration file does not exist.
        SimpleConfigError: If there's a circular dependency or the file format is invalid.
    """
    if seen is None:
        seen = set()

    path = Path(path).resolve()

    if not path.exists():
        raise ConfigNotFoundError(f"Config file not found: {path}")

    if path in seen:
        raise SimpleConfigError(f"Circular dependency detected in config files: {' -> '.join(str(p) for p in seen)} -> {path}")

    new_seen = seen.copy()
    new_seen.add(path)

    try:
        parser = ParserRegistry.get_parser(str(path))
        data = parser.load(str(path))
    except Exception as e:
        raise SimpleConfigError(f"Failed to load config file '{path}': {str(e)}") from e

    if not isinstance(data, dict):
        raise SimpleConfigError(f"Config from '{path}' must be a dictionary, got {type(data).__name__}")

    # Step 3: Resolve inheritance recursively
    bases = data.pop(inheritance_key, [])
    if isinstance(bases, str):
        bases = [bases]
    elif not isinstance(bases, list):
        raise SimpleConfigError(f"The '{inheritance_key}' key in '{path}' must be a string or a list of strings, got {type(bases).__name__}")

    result = {}
    for base_rel_path in bases:
        if not isinstance(base_rel_path, str):
            raise SimpleConfigError(f"Inheritance paths must be strings, got {type(base_rel_path).__name__} in '{path}'")
        base_path = (path.parent / base_rel_path).resolve()
        base_config = load_raw_config(base_path, seen=new_seen, inheritance_key=inheritance_key)
        result = merge_dicts(result, base_config)

    # Finally merge this config over the bases
    result = merge_dicts(result, data)

    return result

class ConfigNotFoundException(Exception):
    pass
