from ml_runner.core.config import loader as cl
from ml_runner.core.config.schema import RunConfig
from ml_runner.core.config.utils import merge_dicts
from ml_runner.core.extensions.extension_manager import ExtensionManager
from ml_runner.core.runner import run_job
from ml_runner.core.cli_interface.cli_registry import CLIRegistry
from ml_runner.core.cli_interface.cli_argument import CLIArgument
from ml_runner.core.exporters.manager import ExporterManager

# Choose which parser backend to use:
from ml_runner.core.cli_interface.arparse_parser import ArgparseParser as Parser
# from ml_runner.core.cli_interface.typer_parser import TyperParser as Parser  # Alternative cli Parser

def register_arguments(ext_manager: ExtensionManager, exporter_manager: ExporterManager):
    """
    Centrally defines core arguments and allows extensions to register theirs.
    This is where help messages and argument definitions are maintained.
    """
    # -------------------------
    # Core Positional Arguments
    # -------------------------
    CLIRegistry.register(CLIArgument("operation", help="Operation to perform",
                                choices=["batch", "run", "stats", "export"]))

    # -------------------------
    # Configuration Arguments
    # -------------------------
    CLIRegistry.register(CLIArgument("-c", dest="config", action="append",
                                help="Path to config files. Can be used multiple times."))

    # -------------------------
    # Common Config Overrides
    # -------------------------
    CLIRegistry.register(CLIArgument("--device", action="append", help="Force device (e.g. cuda, cpu, mps)",
                                config_path="device"))
    CLIRegistry.register(CLIArgument("--lr", type=float, help="Override learning rate",
                                config_path="optimizer.lr"))
    CLIRegistry.register(CLIArgument("--batch-size", type=int, help="Override batch size",
                                config_path="dataloader.batch_size"))
    CLIRegistry.register(CLIArgument("--eval-batch-size", type=int, help="Override evaluation batch size",
                                config_path="evaluation.batch_size"))
    CLIRegistry.register(CLIArgument("--epochs", type=int, help="Override number of training epochs",
                                config_path="training.epochs"))

    # -------------------------
    # Training & Evaluation Flags
    # -------------------------
    CLIRegistry.register(CLIArgument("--train", action="store_true", dest="do_train",
                                help="Enable training", config_path="training.enabled"))
    CLIRegistry.register(CLIArgument("--dont-train", action="store_false", dest="do_train",
                                help="Disable training", config_path="training.enabled"))

    CLIRegistry.register(CLIArgument("--eval", action="store_true", dest="do_eval",
                                help="Enable evaluation after training", config_path="evaluation.enabled"))
    CLIRegistry.register(CLIArgument("--dont-eval", action="store_false", dest="do_eval",
                                help="Disable evaluation", config_path="evaluation.enabled"))

    # -------------------------
    # Logging
    # -------------------------
    CLIRegistry.register(CLIArgument("--log-level", help="Set logging level (DEBUG, INFO, etc.)",
                                config_path="logging.level"))

    # -------------------------
    # Extension Overrides
    # -------------------------
    ext_manager.register_cli_arguments(CLIRegistry)
    # Allow exporters to register CLI args
    exporter_manager.register_cli_arguments(CLIRegistry)

    # -------------------------
    # Exporter CLI helpers
    # -------------------------
    # Positional exporter name (optional) used by `ml-runner export <exporter>`
    CLIRegistry.register(CLIArgument("exporter", help="Exporter name to run (e.g. data, onnx)", nargs='?'))


def load_config(configs, overrides, ext_manager: ExtensionManager, exporter_manager: ExporterManager) -> RunConfig:
    """Loads and merges configuration files and CLI overrides."""
    cfg_dict = {}
    if configs:
        for cfile in configs:
            new_cfg = cl.load_config(cfile)
            cfg_dict = merge_dicts(cfg_dict, new_cfg)

    # Apply CLI overrides
    cfg_dict = merge_dicts(cfg_dict, overrides)

    # Create RunConfig with extension config classes
    # Include exporter config classes so they can be parsed from the config under
    # the top-level `exports` key.
    # Build RunConfig using extension config classes and exporter config classes
    exporter_config_classes = exporter_manager.get_config_classes()
    run_cfg = RunConfig.from_dict(cfg_dict, extension_config_classes=ext_manager.get_config_classes(), exporter_config_classes=exporter_config_classes)

    return run_cfg


def main():
    ext_manager = ExtensionManager()
    ext_manager.register_extensions()

    exporter_manager = ExporterManager()
    exporter_manager.register_exporters()

    # Setup CLI registry
    CLIRegistry.clear()
    register_arguments(ext_manager, exporter_manager)

    # Initialize Parser (Argparse or Typer)
    parser = Parser(CLIRegistry)

    # Parse arguments and collect config overrides
    args = parser.parse_args()
    overrides = parser.get_config_overrides()

    op = args.get('operation')

    if op == 'run':
        # Single run
        configs = args.get('config', [])
        run_cfg = load_config(configs, overrides, ext_manager, exporter_manager)

        # print(f"configs: {load_config(configs, {}, ext_manager)}")

        # print(f"run_cfg: {run_cfg}")

        run_job(run_cfg, args, ext_manager)

    elif op == 'batch':
        # Batch run over multiple config files
        configs = args.get('config', [])
        for config in configs:
            run_cfg = load_config([config], overrides, ext_manager)
            run_job(run_cfg, args, ext_manager)

    elif op == 'stats':
        print("Statistics not yet implemented")
    elif op == 'export':
        # Export dispatcher: single-export or config-driven
        exporter_name = args.get('exporter')

        if not exporter_name:
            print("Available exporters:", exporter_manager.all())
            return

        # Build global RunConfig so exporters get access to the full configuration.
        configs = args.get('config', [])
        run_cfg = load_config(configs, overrides, ext_manager, exporter_manager)

        # Run the exporter via ExporterManager (passes exporter-specific config and full run_cfg)
        exporter_manager.run_exporter(exporter_name, run_cfg)
    else:
        raise ValueError(f"Unknown operation: {op}")


if __name__ == "__main__":
    main()
