import yaml

from registries import ConfigRegistry

@ConfigRegistry.register(".yaml", ".yml")
def parse_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)
