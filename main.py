import argparse
import torch
import os
import sys
import glob
from torch.utils.data import DataLoader
from pathlib import Path

# Add src to sys.path to allow importing ml_runner
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from ml_runner.core.engine.engine import Engine
from ml_runner.core.log_utils.logger_setup import setup_logging

import ml_runner.core.config.loader as cl
from ml_runner.core.config.run_config import RunConfig
from ml_runner.core.config.path_utils import resolve_path_template, ensure_dir
from ml_runner.core.log_utils import logger
# import ml_runner.extensions as extensions # Bootstrap all extensions early

from ml_runner.core.tasks.classification_task import ClassificationTask

from ml_runner.core.registries import ModelRegistry, DatasetRegistry, OptimizerRegistry, LossRegistry, MetricRegistry, resolve_component

from ml_runner.core.callbacks.time_profiler import TimeProfiler
# from ml_runner.core.callbacks.checkpoint_callback import CheckpointCallback
from ml_runner.core.callbacks.evaluation_callback import EvaluationCallback
from ml_runner.core.callbacks.batch_logger import BatchLogger
from ml_runner.core.callbacks.epoch_logger import EpochLogger
from ml_runner.core.callbacks.eval_logger import EvalLogger

# Import components to register them
# import ml_runner.builtins.datasets.common_datasets
# import ml_runner.builtins.models.mlp
# import ml_runner.builtins.models.cnn
# import ml_runner.builtins.models.rnn

# Load built-in extensions
import ml_runner.extensions


def setup_argparse():
    parser = argparse.ArgumentParser(description="ML Job Runner - Refactored")
    parser.add_argument("operation",
                        choices=['batch', 'run', 'vis', 'stats'],
                        help="Operation to run")

    # Config loading
    parser.add_argument("--config", "-c", action="append", help="Config file(s) to load")
    parser.add_argument("--log-level", dest="log_level", help="Log level (DEBUG, INFO, WARNING, ERROR)")
    parser.add_argument("--log-dir", dest="log_dir", help="Output directory for logs")

    # Behavior Flags (nested mapping)
    parser.add_argument("--train", action="store_true", dest="do_train", default=None, help="Enable training")
    parser.add_argument("--dont-train", action="store_false", dest="do_train", help="Disable training")
    parser.add_argument("--eval", action="store_true", dest="do_eval", default=None, help="Enable evaluation")
    parser.add_argument("--dont-eval", action="store_false", dest="do_eval", help="Disable evaluation")

    # Model and Hyperparameters (mapped to nested structure)
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--device", help="Override device (cpu, cuda, auto)")

    # Checkpoint options
    parser.add_argument("--checkpoint-dir", dest="checkpoint_dir", help="Override checkpoint directory")
    parser.add_argument("--checkpoint-frequency", type=int, help="Override checkpoint frequency")

    return parser


def apply_overrides(cfg_dict, args):
    """
    Apply CLI arguments to the config dictionary, handling nested structures.
    """
    if args.lr is not None:
        if "optimizer" not in cfg_dict: cfg_dict["optimizer"] = {}
        opt_key = next(iter(cfg_dict["optimizer"])) if cfg_dict["optimizer"] else "adam"
        if opt_key not in cfg_dict["optimizer"]: cfg_dict["optimizer"][opt_key] = {}
        cfg_dict["optimizer"][opt_key]["lr"] = args.lr

    if args.batch_size is not None:
        if "dataloader" not in cfg_dict: cfg_dict["dataloader"] = {}
        cfg_dict["dataloader"]["batch_size"] = args.batch_size

    if args.epochs is not None:
        if "training" not in cfg_dict: cfg_dict["training"] = {}
        cfg_dict["training"]["epochs"] = args.epochs

    if args.device is not None:
        cfg_dict["device"] = args.device

    if args.checkpoint_dir is not None:
        if "checkpoint" not in cfg_dict: cfg_dict["checkpoint"] = {}
        cfg_dict["checkpoint"]["directory"] = args.checkpoint_dir

    if args.checkpoint_frequency is not None:
        if "checkpoint" not in cfg_dict: cfg_dict["checkpoint"] = {}
        cfg_dict["checkpoint"]["frequency"] = args.checkpoint_frequency

    if args.do_train is not None:
        if "training" not in cfg_dict: cfg_dict["training"] = {}
        cfg_dict["training"]["enabled"] = args.do_train

    if args.do_eval is not None:
        if "eval" not in cfg_dict: cfg_dict["eval"] = {}
        cfg_dict["eval"]["enabled"] = args.do_eval

    if args.log_level is not None:
        if "logging" not in cfg_dict: cfg_dict["logging"] = {}
        cfg_dict["logging"]["level"] = args.log_level

    if args.log_dir is not None:
        if "logging" not in cfg_dict: cfg_dict["logging"] = {}
        cfg_dict["logging"]["output_dir"] = args.log_dir

    return cfg_dict


def build_runtime(run_cfg: RunConfig):
    # Resolve Device
    if run_cfg.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = run_cfg.device

    # Build Datasets
    dataset_cls = DatasetRegistry.get(run_cfg.dataset.name)
    train_ds = dataset_cls(**run_cfg.dataset.params, train=True)
    val_ds = dataset_cls(**run_cfg.dataset.params, train=False)

    # Build Loaders
    dl_params = run_cfg.dataloader.copy()

    # Use training.batch_size if present and dataloader batch_size is not
    if "batch_size" not in dl_params and run_cfg.training.batch_size:
        dl_params["batch_size"] = run_cfg.training.batch_size

    shuffle = dl_params.pop("shuffle", run_cfg.training.shuffle)
    train_loader = DataLoader(train_ds, **dl_params, shuffle=shuffle)
    val_loader = DataLoader(val_ds, batch_size=dl_params.get("batch_size", 64))

    # Build Model
    model_cls = ModelRegistry.get(run_cfg.model.name)
    model = model_cls(**run_cfg.model.params).to(device)

    # Build Loss
    loss_cls = LossRegistry.get(run_cfg.loss.name)
    loss_fn = loss_cls(**run_cfg.loss.params)

    # Build Optimizer
    optimizer_cls = OptimizerRegistry.get(run_cfg.optimizer.name)
    optimizer = optimizer_cls(model.parameters(), lr=run_cfg.optimizer.lr, **run_cfg.optimizer.params)

    # Build Metrics
    metrics = [MetricRegistry.get(m) for m in run_cfg.metrics]

    # Build Task
    task = ClassificationTask(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        metrics=metrics
    )

    return task, device


def run_job(run_cfg: RunConfig, args):
    from ml_runner.core.log_utils.logger_setup import setup_logging
    import logging

    # -------------------------------
    # Path Resolution & Dir Creation
    # -------------------------------
    context = {
        "experiment_name": run_cfg.experiment.name,
        "device": run_cfg.device
    }

    log_file_name = resolve_path_template(run_cfg.logging.log_file, context)
    log_dir = resolve_path_template(run_cfg.logging.output_dir, context)
    log_path = str(Path(log_dir) / log_file_name)
    ensure_dir(log_path)

    # Reset logging handlers to avoid duplicates in batch mode
    root_logger = logging.getLogger("mltool")
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    job_logger = setup_logging(level=run_cfg.logging.level, log_file=log_path)
    job_logger.info(f"Starting Job: {run_cfg.experiment.name}")

    # Build Task & Device
    task, device = build_runtime(run_cfg)

    # -------------------------------
    # Build Callbacks
    # -------------------------------
    callbacks = []
    callbacks.append(EpochLogger())
    callbacks.append(EvalLogger())

    if run_cfg.evaluation.eval_during_training:
        callbacks.append(EvaluationCallback(every_n_epochs=run_cfg.evaluation.eval_frequency))

    from ml_runner.core.registries import ExtensionRegistry

    for ext_name in ExtensionRegistry.all():
        ext_cls = ExtensionRegistry.get(ext_name)
        ext = ext_cls()
        callbacks.extend(ext.create_callbacks(run_cfg))

    # Create Engine
    engine = Engine(task=task, callbacks=callbacks)

    # -------------------------------
    # Execution
    # -------------------------------
    if run_cfg.do_train:
        job_logger.info(f"Training for {run_cfg.training.epochs} epochs")
        engine.train(epochs=run_cfg.training.epochs)

    if run_cfg.do_eval:
        if run_cfg.evaluation.eval_checkpoints and not run_cfg.do_train:
            # Standalone evaluation of checkpoints
            checkpoint_dir = resolve_path_template(run_cfg.checkpoint.directory, context)
            checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
            if not checkpoints:
                job_logger.warning(f"No checkpoints found in {checkpoint_dir} for evaluation")
                engine.evaluate()
            else:
                job_logger.info(f"Found {len(checkpoints)} checkpoints to evaluate")
                for ckpt in sorted(checkpoints):
                    job_logger.info(f"Evaluating checkpoint: {ckpt}")
                    task.load_checkpoint(ckpt)

                    # Sync engine epoch with checkpoint if possible
                    import re
                    match = re.search(r"epoch_(\d+)", Path(ckpt).name)
                    if match:
                        engine.eval_state.epoch = int(match.group(1))

                    engine.evaluate()
        else:
            job_logger.info("Running standard evaluation")
            engine.evaluate()

    job_logger.info(f"Job {run_cfg.experiment.name} Complete")


def main():
    parser = setup_argparse()
    args = parser.parse_args()

    if args.operation == 'batch':
        if not args.config:
            print("Error: --config required for batch mode")
            return

        for cfile in args.config:
            print(f"\n--- Processing Config: {cfile} ---")
            cfg_dict = cl.load_config(cfile)
            cfg_dict = apply_overrides(cfg_dict, args)
            run_cfg = RunConfig.from_dict(cfg_dict)
            run_job(run_cfg, args)

    else:
        # Single Run Mode (run, vis, stats): merge all configs
        cfg_dict = {}
        if args.config:
            for cfile in args.config:
                new_cfg = cl.load_config(cfile)
                cfg_dict = cl.merge_dicts(cfg_dict, new_cfg)

        cfg_dict = apply_overrides(cfg_dict, args)
        run_cfg = RunConfig.from_dict(cfg_dict)

        if args.operation == 'run':
            run_job(run_cfg, args)
        elif args.operation == 'vis':
            print("Visualizations not fully implemented")
        elif args.operation == 'stats':
            print("Statistics not fully implemented")


if __name__ == "__main__":
    main()
