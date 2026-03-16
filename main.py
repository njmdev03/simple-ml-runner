import torch
from torch.utils.data import DataLoader

from engine.engine import Engine
from log_utils.logger_setup import setup_logging

import config.loader as cl

from tasks.classification_task import ClassificationTask

from registries import ModelRegistry
from registries import DatasetRegistry
from registries import OptimizerRegistry
from registries import LossRegistry
from registries import MetricRegistry
from registries import resolve_component

from callbacks.time_profiler import TimeProfiler
from callbacks.checkpoint_callback import CheckpointCallback
from callbacks.evaluation_callback import EvaluationCallback
from callbacks.batch_logger import BatchLogger
from callbacks.epoch_logger import EpochLogger
from callbacks.eval_logger import EvalLogger

from datasets import common_datasets


def main():
    # -------------------------------
    # Setup Logging
    # -------------------------------

    logger = setup_logging(level="INFO", log_file="train.log")

    logger.info("Starting ML experiment")

    # -------------------------------
    # Load Config
    # -------------------------------

    cfg = cl.load_config("./examples/MNIST/mnist_mlp.yml")

    # -------------------------------
    # Resolve Device
    # -------------------------------

    device = "cuda" if torch.cuda.is_available() and "cuda" in cfg["device"] else "cpu"

    # -------------------------------
    # Build DataLoaders
    # -------------------------------

    dataset_cls = resolve_component(cfg["dataset"], DatasetRegistry)

    params = cfg["dataset"][cfg["dataset"]]

    val_dataset = dataset_cls.__init__(params)

    params["train"] = True

    train_dataset = dataset_cls.__init__(params)

    batch_size = cfg["dataloader"]["batch_size"]
    num_workers = None
    pin_memory = False
    collate_fn = None
    shuffle = cfg["dataloader"].get("shuffle", True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        shuffle=shuffle
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )

    # -------------------------------
    # Build Model
    # -------------------------------

    model_cls = resolve_component(cfg["model"], ModelRegistry)

    model = model_cls(cfg["model"][cfg["model"]])

    # -------------------------------
    # Build Loss
    # -------------------------------

    loss_fn_cls = resolve_component(cfg["loss"], LossRegistry)

    loss_fn = loss_fn_cls(cfg["loss"][cfg["loss"]])

    # -------------------------------
    # Build Optimizer
    # -------------------------------

    optimizer_cls = resolve_component(cfg["optimizer"], OptimizerRegistry)

    optimizer = optimizer_cls(model.parameters(), cfg["optimizer"][cfg["optimizer"]])

    # -------------------------------
    # Build Metrics
    # -------------------------------

    metrics = []

    for metric_name in cfg.get("metrics", []):
        metric_fn = MetricRegistry.get(metric_name)
        metrics.append(metric_fn)

    # -------------------------------
    # Build Task
    # -------------------------------

    task = ClassificationTask(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        metrics=metrics
    )

    # -------------------------------
    # Build Callbacks
    # -------------------------------

    # callbacks = build_callbacks(cfg)

    callbacks = []

    callbacks.append(EpochLogger())

    if cfg["evaluation"]["eval_during_training"] and cfg["evaluation"]["eval_checkpoints"]:
        callbacks.append(EvaluationCallback(every_n_epochs=cfg["evaluation"]["eval_frequency"]))

    callbacks.append(EvalLogger())

    # callbacks.append(BatchLogger(every_n_batches=10, every_n_eval_batches=10))

    callbacks.append(CheckpointCallback(path=cfg["checkpoint"]["directory"], every_n_epochs=cfg["checkpoint"]["frequency"]))

    callbacks.append(TimeProfiler())

    # -------------------------------
    # Create Engine
    # -------------------------------

    engine = Engine(
        task=task,
        callbacks=callbacks
    )

    # -------------------------------
    # Run Training
    # -------------------------------

    epochs = cfg["training"]["epochs"]

    try:
        engine.train(epochs=epochs)
    except KeyboardInterrupt:
        logger.warning("User interrupted training")

    logger.info("Training complete")

    # -------------------------------
    # 13. Run Evaluation
    # -------------------------------

    try:
        engine.evaluate()
    except KeyboardInterrupt:
        logger.warning("User interrupted evaluation")

    logger.info("Job Complete")


if __name__ == "__main__":
    main()