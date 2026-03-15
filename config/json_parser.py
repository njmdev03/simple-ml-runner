import json

from registries.config_registry import ConfigRegistry


@ConfigRegistry.register(".json")
def parse_json(path):
    with open(path) as f:
        return json.load(f)
