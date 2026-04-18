from pathlib import Path
from typing import Union, Optional, Set, Dict
from simple_config.parser.registry import ParserRegistry
from simple_config.utils import merge_dicts


class ConfigLoader:
    # _cache: Dict[Path, dict] = {}

    # @classmethod
    # def clear_cache(cls):
    #     cls._cache = {}

    @classmethod
    def _resolve_to_dict(cls, config: Path) -> dict:
        """Reads a file path to a dict. Will also read from a cache of file paths that have already been resolved.

        Args:
            config: string or pathlib path to a config file to be loaded to a dict. Files are resolved to dicts using
            the ParserRegistry or cached resolutions.

        Returns:
            dict: the loaded config dictionary
        """
        # if cls._cache.get(config):
        #     return cls._cache.get(config)

        if not config.exists():
            raise ConfigNotFoundException

        if not ParserRegistry.get_parser(config):
            raise UnsupportedConfigException

        return ParserRegistry.get_parser(config).load(config)

    @classmethod
    def _load_config(
            cls,
            config: Union[Dict, Path, str],
            # seen: Optional[Set[Path]] = [],
            inheritance_key: str = "config"
        ) -> dict:
        # new_seen = seen

        # Resolve string to a path
        if isinstance(config, str):
            config = Path(config)

        # Resolve Path to a dict
        if isinstance(config, Path):
            top_dict = cls._resolve_to_dict(config)

            # Save explored paths for circular dependency checks.
            # new_seen.append(config)
        else:
            top_dict = config

        inherits = top_dict.get(inheritance_key)

        if inherits:
            top_dict.pop(inheritance_key)

            base_dict = cls._load_multi_config(*inherits, inheritance_key=inheritance_key)

            return merge_dicts(base_dict, top_dict)
        else:
            return top_dict

    @classmethod
    def _load_multi_config(
        cls,
        *configs: Union[Dict, Path, str],
        inheritance_key: str = "config"
        ) -> dict:
        result = {}

        for config in configs:
            merge_dicts(result, cls._load_config(config, inheritance_key=inheritance_key))

        return result

    @classmethod
    def load_config(
        cls,
        *config: Union[Dict, Path, str],
        inheritance_key: str = "config"
        ) -> dict:
        return cls._load_multi_config(*config, inheritance_key=inheritance_key)

class ConfigNotFoundException(Exception):
    pass

class UnsupportedConfigException(Exception):
    pass

class CircularDependencyException(Exception):
    pass
