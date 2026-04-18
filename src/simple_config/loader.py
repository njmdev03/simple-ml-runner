from pathlib import Path
from typing import Union, Optional, Set, Dict
from simple_config.parser.registry import ParserRegistry
from simple_config.utils import merge_dicts


class ConfigLoader:
    # _cache: Dict[Path, dict] = {}

    # @classmethod
    # def clear_cache(self):
    #     self._cache = {}

    def __init__(self, parser_reg: ParserRegistry):
        self._parser_reg = parser_reg

    def _resolve_to_dict(self, config: Path) -> dict:
        """Reads a file path to a dict. Will also read from a cache of file paths that have already been resolved.

        Args:
            config: string or pathlib path to a config file to be loaded to a dict. Files are resolved to dicts using
            the ParserRegistry or cached resolutions.

        Returns:
            dict: the loaded config dictionary
        """
        # if self._cache.get(config):
        #     return self._cache.get(config)

        if not config.exists():
            raise ConfigNotFoundException

        if not self._parser_reg.get_parser(config):
            raise UnsupportedConfigException

        return self._parser_reg.get_parser(config).load(config)

    def _load_config(
            self,
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
            top_dict = self._resolve_to_dict(config)

            # Save explored paths for circular dependency checks.
            # new_seen.append(config)
        else:
            top_dict = config

        inherits = top_dict.get(inheritance_key)

        if inherits:
            top_dict.pop(inheritance_key)

            base_dict = self._load_multi_config(*inherits, inheritance_key=inheritance_key)

            return merge_dicts(base_dict, top_dict)
        else:
            return top_dict

    def _load_multi_config(
        self,
        *configs: Union[Dict, Path, str],
        inheritance_key: str = "config"
        ) -> dict:
        result = {}

        for config in configs:
            merge_dicts(result, self._load_config(config, inheritance_key=inheritance_key))

        return result

    def load_config(
        self,
        *config: Union[Dict, Path, str],
        inheritance_key: str = "config"
        ) -> dict:
        return self._load_multi_config(*config, inheritance_key=inheritance_key)

class ConfigNotFoundException(Exception):
    pass

class UnsupportedConfigException(Exception):
    pass

class CircularDependencyException(Exception):
    pass
