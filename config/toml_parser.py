import tomllib

from registries import ConfigRegistry

@ConfigRegistry.register(".toml")
def parse_toml(path):
    with open(path, "rb") as f:
        return tomllib.load(f)
