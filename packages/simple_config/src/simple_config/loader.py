from pathlib import Path
from typing import Union, List, Dict
from copy import deepcopy, copy

from simple_config.parser.registry import ParserRegistry
from simple_config.utils import merge_dicts


class ConfigLoader:
    """Loads and merges configuration files with inheritance support.

    Attributes:
        _parser_reg: Registry for configuration parsers.
        _use_cache: Whether to cache loaded configurations.
        _cache: Dictionary mapping paths to cached configuration dicts.
    """

    def __init__(self, parser_reg: ParserRegistry, cache_configs: bool = False):
        """Initializes ConfigLoader.

        Args:
            parser_reg: Registry to use for parsing files.
            cache_configs: If True, cache results of file resolutions. Defaults to False.
        """
        self._parser_reg: ParserRegistry = parser_reg
        self._use_cache: bool = cache_configs
        self._cache: Dict[Path, dict] = {}

    def _resolve_to_dict(self, config: Path) -> dict:
        """Resolves file path to dictionary using registered parsers.

        Args:
            config: Path to the configuration file.

        Returns:
            dict: The loaded configuration dictionary.

        Raises:
            ConfigNotFoundException: If the file does not exist.
            UnsupportedConfigException: If no parser is registered for the file extension.
        """
        if self._use_cache and self._cache.get(config):
            return self._cache.get(config)

        if not config.exists():
            raise ConfigFileNotFoundError(f"Config file not found: {config}")

        if not self._parser_reg.get_parser(config):
            raise ConfigParserMissingError(f"No parser registered for: {config.suffix}")

        return self._parser_reg.get_parser(config).load(config)

    def _load_config(
            self,
            config: Union[Dict, Path, str],
            seen: List[Path] = [],
            inheritance_key: str = "config"
        ) -> dict:
        """Loads a single configuration and recursively merges base configs.

        Args:
            config: Configuration dictionary, file path, or path string.
            seen: List of paths already visited to detect circular dependencies.
            inheritance_key: Key in dictionary used for base config paths. Defaults to "config".

        Returns:
            dict: The merged configuration dictionary.

        Raises:
            CircularDependencyException: If a circular inheritance is detected.
        """
        new_seen = copy(seen)

        # Resolve string to a path
        if isinstance(config, str):
            config = Path(config)

        # Resolve Path to a dict
        if isinstance(config, dict):
            top_dict = deepcopy(config)
        else:
            if config in seen:
                raise CircularInheritanceError(f"Circular inheritance: {' -> '.join(map(str, seen + [config]))}")

            new_seen.append(config)
            top_dict = deepcopy(self._resolve_to_dict(config))

        inherits = top_dict.get(inheritance_key)

        if inherits:
            top_dict.pop(inheritance_key)

            base_dict = self._load_multi_config(*inherits, seen=new_seen, inheritance_key=inheritance_key)

            result = merge_dicts(base_dict, top_dict)
        else:
            result = top_dict

        if isinstance(config, Path):
            self._cache[config] = top_dict

        return result

    def _load_multi_config(
        self,
        *configs: Union[Dict, Path, str],
        seen: List[Path] = [],
        inheritance_key: str = "config"
        ) -> dict:
        """Loads and merges multiple configurations.

        Args:
            *configs: Variadic configuration dictionaries, file paths, or path strings.
            seen: List of paths already visited.
            inheritance_key: Key for inheritance. Defaults to "config".

        Returns:
            dict: The combined configuration dictionary.
        """
        result = {}

        for config in configs:
            merge_dicts(result, self._load_config(config, seen=seen, inheritance_key=inheritance_key))

        return result

    def load_config(
        self,
        *config: Union[Dict, Path, str],
        inheritance_key: str = "config"
        ) -> dict:
        """Public API to load and merge configurations.

        Args:
            *config: Variadic configuration sources.
            inheritance_key: Key for inheritance. Defaults to "config".

        Returns:
            dict: The final merged configuration dictionary.
        """
        return self._load_multi_config(*config, inheritance_key=inheritance_key)

class ConfigLoaderError(Exception):
    """Base class for ConfigLoader exceptions."""
    pass

class ConfigFileNotFoundError(ConfigLoaderError):
    """Raised when a config file does not exist."""
    pass

class ConfigParserMissingError(ConfigLoaderError):
    """Raised when no parser exists for a file extension."""
    pass

class CircularInheritanceError(ConfigLoaderError):
    """Raised when circular inheritance is detected."""
    pass
