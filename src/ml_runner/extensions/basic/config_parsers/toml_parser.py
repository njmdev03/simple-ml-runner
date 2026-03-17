import tomllib

from ml_runner.core.registries import ConfigParser

@ConfigParser(".toml")
def parse_toml(path):
    with open(path, "rb") as f:
        return tomllib.load(f)
