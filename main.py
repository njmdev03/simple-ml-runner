import argparse
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from engine.engine import Engine
from log_utils.logger_setup import setup_logging
from callbacks.time_profiler import TimeProfiler
from callbacks.checkpoint_callback import CheckpointCallback
from callbacks.evaluation_callback import EvaluationCallback
from callbacks.batch_logger import BatchLogger
from callbacks.epoch_logger import EpochLogger
from callbacks.eval_logger import EvalLogger

import config.loader as cl

from tasks.classification_task import ClassificationTask


def accuracy(outputs, targets):
    preds = outputs.argmax(dim=1)
    return (preds == targets).float().mean().item()


def main():
    # -------------------------------
    # Parse CLI Arguments
    # -------------------------------
    parser = argparse.ArgumentParser(description="ML Job Runner")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the config file (yaml, json, toml, ini)"
    )
    args = parser.parse_args()
    config_path = Path(args.config).resolve()

    # -------------------------------
    # Load Config
    # -------------------------------
    cfg = cl.load_config(config_path)
    # TODO: Config to object

    # -------------------------------
    # Setup Logging
    # -------------------------------
    # TODO: Switch to config object
    log_file = cfg.get("log_file", "train.log")
    log_level = cfg.get("log_level", "INFO")
    logger = setup_logging(level=log_level, log_file=log_file)
    logger.info(f"Starting ML experiment using config {config_path}")

    print(cfg)

    quit()


    # -------------------------------
    # Load Data
    # -------------------------------
    train_dataset = MNIST("./data/", download=True, transform=ToTensor())
    val_dataset = MNIST("./data/", train=False, download=True, transform=ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    # -------------------------------
    # Define Model + Loss + Task
    # -------------------------------
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    task = ClassificationTask(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device= "cuda" if torch.cuda.is_available() else "cpu",
        metrics=[
            accuracy
        ]  # Add metric functions if needed
    )

    # -------------------------------
    # Setup Callbacks
    # -------------------------------
    callbacks = []

    callbacks.append(EpochLogger())

    callbacks.append(EvaluationCallback(every_n_epochs=1))

    callbacks.append(EvalLogger())

    callbacks.append(BatchLogger(every_n_batches=10, every_n_eval_batches=10))

    callbacks.append(CheckpointCallback(path="checkpoints/", every_n_epochs=1))

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
    try:
        engine.train(epochs=5)
    except KeyboardInterrupt:
        logger.warning(f"User interrupted training")

    logger.info("Training complete")

    # -------------------------------
    # Optionally Run Evaluation Only
    # -------------------------------
    logger.info("Running final evaluation with validation set")
    try:
        engine.evaluate()
    except KeyboardInterrupt:
        logger.warning(f"User interrupted Evaluation")

    logger.info("Job Complete")


if __name__ == "__main__":
    main()