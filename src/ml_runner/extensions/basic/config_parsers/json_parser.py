import json

from ml_runner.core.registries import ConfigParser

@ConfigParser(".json")
def parse_json(path):
    with open(path) as f:
        return json.load(f)
