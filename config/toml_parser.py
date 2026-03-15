import tomllib

from registries.config_registry import ConfigRegistry

@ConfigRegistry.register(".toml")
def parse_toml(path):
    with open(path, "rb") as f:
        return tomllib.load(f)