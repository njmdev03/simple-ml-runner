import argparse
from dataclasses import dataclass

from simple_config import ConfigBuilder, Variant
from simple_registries import Registry


def get_schema(model_registry: Registry):
    @dataclass
    class ExperimentSchema:
        name: str

    @dataclass
    class Schema:
        experiment: ExperimentSchema
        model: Variant = Variant({key: model_registry.get(key) for key in model_registry.keys()})

    return Schema

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("op", type=str, default="run", choices=["run", "batch", "export"])

    parser.add_argument("--config", type=str, required=False)

    args = parser.parse_args()



    conf_builder = ConfigBuilder(get_schema())

    if args.op == "run":
        print("Run")

    elif args.op == "batch":
        print("Run Batch")

    elif args.op == "export":
        print("Run one-shot task")

    else:
        raise ValueError("Invalid operation requested")

if __name__ == "__main__":
    main()