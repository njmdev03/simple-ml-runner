import argparse
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from engine.engine import Engine
from log_utils.logger_setup import setup_logging

import config.loader as cl
from config.run_config import RunConfig

from tasks.classification_task import ClassificationTask

from registries import ModelRegistry, DatasetRegistry, OptimizerRegistry, LossRegistry, MetricRegistry, resolve_component

from callbacks.time_profiler import TimeProfiler
from callbacks.checkpoint_callback import CheckpointCallback
from callbacks.evaluation_callback import EvaluationCallback
from callbacks.batch_logger import BatchLogger
from callbacks.epoch_logger import EpochLogger
from callbacks.eval_logger import EvalLogger

# Import components to register them
import datasets.common_datasets
import models.mlp
import models.cnn
import models.rnn


def setup_argparse():
    parser = argparse.ArgumentParser(description="ML Job Runner - Refactored")
    parser.add_argument("operation",
                        choices=['batch', 'job', 'test', 'train', 'vis', 'stats'],
                        help="Operation to run")

    # Config loading
    parser.add_argument("--config", "-c", action="append", help="Config file(s) to load")
    parser.add_argument("--silent", action="store_true", help="Print less to console")

    # Behavior Flags (nested mapping)
    parser.add_argument("--train", action="store_true", dest="do_train", default=True, help="Enable training")
    parser.add_argument("--dont-train", action="store_false", dest="do_train", help="Disable training")
    parser.add_argument("--test", action="store_true", dest="do_test", default=True, help="Enable testing")
    parser.add_argument("--dont-test", action="store_false", dest="do_test", help="Disable testing")

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
        # Assuming optimizer is at the root and has a name key containing the params
        # This is a bit tricky since the new format is nested like: optimizer: { adam: { lr: ... } }
        # We'll find the first key in optimizer and update its lr if it exists
        if "optimizer" in cfg_dict:
            opt_key = next(iter(cfg_dict["optimizer"]))
            cfg_dict["optimizer"][opt_key]["lr"] = args.lr
        else:
            cfg_dict["optimizer"] = {"adam": {"lr": args.lr}}

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


def main():
    parser = setup_argparse()
    args = parser.parse_args()

    # -------------------------------
    # Setup Logging
    # -------------------------------
    log_level = "WARNING" if args.silent else "INFO"
    logger = setup_logging(level=log_level, log_file="train.log")
    logger.info(f"Starting ML experiment - Operation: {args.operation}")

    # -------------------------------
    # Load and Merge Configs
    # -------------------------------
    cfg_dict = {}
    if args.config:
        for cfile in args.config:
            cfg_dict = cl.load_config(cfile) # This already handles 'extends' and merges

    # Apply CLI Overrides
    cfg_dict = apply_overrides(cfg_dict, args)

    # Build typed Config
    run_cfg = RunConfig.from_dict(cfg_dict)

    # -------------------------------
    # Build Task & Device
    # -------------------------------
    task, device = build_runtime(run_cfg)

    # -------------------------------
    # Build Callbacks
    # -------------------------------
    callbacks = []
    callbacks.append(EpochLogger())

    if run_cfg.evaluation.eval_during_training:
        callbacks.append(EvaluationCallback(every_n_epochs=run_cfg.evaluation.eval_frequency))

    callbacks.append(EvalLogger())

    if run_cfg.checkpoint.frequency > 0:
        callbacks.append(CheckpointCallback(
            path=run_cfg.checkpoint.directory,
            every_n_epochs=run_cfg.checkpoint.frequency
        ))

    if run_cfg.profile.enabled:
        callbacks.append(TimeProfiler())

    # -------------------------------
    # Create Engine
    # -------------------------------
    engine = Engine(task=task, callbacks=callbacks)

    # -------------------------------
    # Execution Logic
    # -------------------------------
    if args.operation == 'train':
        if args.do_train:
            logger.info("Starting Training phase")
            engine.train(epochs=run_cfg.training.epochs)

        if args.do_test:
            logger.info("Starting Evaluation phase")
            engine.evaluate()

    elif args.operation == 'test':
        logger.info("Starting standalone Evaluation")
        engine.evaluate()

    elif args.operation == 'vis':
        logger.info("Visualizations requested (Not fully implemented in this refactor pass)")
        # Placeholder for visualization logic
        pass

    elif args.operation == 'stats':
        logger.info("Statistics requested")
        # Placeholder for stats logic
        pass

    logger.info("Job Complete")


if __name__ == "__main__":
    main()
