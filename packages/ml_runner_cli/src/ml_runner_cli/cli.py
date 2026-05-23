from functools import partial
import argparse
from dataclasses import dataclass, field
from logging import config

from ml_runner_core.config import MLRunnerConfig
from ml_runner_core.extensions import ExtensionManager, ExtensionNamespace
from simple_config import ConfigBuilder, Variant
from simple_registries import Registry


# def get_schema(model_registry: Registry):
#     @dataclass
#     class ExperimentSchema:
#         name: str

#     @dataclass
#     class Schema:
#         experiment: ExperimentSchema
#         model: Variant = Variant({key: model_registry.get(key) for key in model_registry.keys()})

#     return Schema

def main():
    models_namespace = ExtensionNamespace("models")
    
    parsers_namespace = ExtensionNamespace("config.parsers")

    extension_manager = ExtensionManager(models_namespace, parsers_namespace)

    extension_manager.discover_extensions()
    
    # Get basic config parsers
    
    
    # Get configs from extensions
    model_configs = extension_manager.get_configs(models_namespace.name)

    # Setup config schema
    config_schema = MLRunnerConfig

    config_schema.model = field(default_factory=partial(Variant, model_configs))

    builder = ConfigBuilder(config_schema)

    # Parse CLI Args
    parser = argparse.ArgumentParser()

    parser.add_argument("op", type=str, default="run", choices=["run", "batch", "export"])

    parser.add_argument("--config", type=str, required=False)

    args = parser.parse_args()

    if args.op == "run":
        print("Run")

        configs = args.config

        print(f"Configs: {configs}")

        res = builder.build(configs)

        print(f"Config Results: {res}")

    elif args.op == "batch":
        print("Run Batch")

    elif args.op == "export":
        print("Run one-shot task")

    else:
        raise ValueError("Invalid operation requested")

if __name__ == "__main__":
    main()