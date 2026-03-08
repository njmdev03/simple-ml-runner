# main.py
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from engine.engine import Engine
from log_utils.logger_setup import setup_logging
from callbacks.time_profiler import TimeProfiler
from callbacks.console_logger import ConsoleLogger
from callbacks.checkpoint_callback import CheckpointCallback
from callbacks.evaluation_callback import EvaluationCallback

from tasks.classification_task import ClassificationTask


def main():

    # -------------------------------
    # 1. Setup Logging
    # -------------------------------
    logger = setup_logging(
        level="INFO",
        log_file="train.log"
    )

    logger.info("Starting ML experiment")

    # -------------------------------
    # 2. Load Data
    # -------------------------------
    train_dataset = MNIST("./data/", download=True, transform=ToTensor())
    val_dataset = MNIST("./data/", train=False, download=True, transform=ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    # -------------------------------
    # 3. Define Model + Loss + Task
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

        ]  # Add metric functions if needed
    )

    # -------------------------------
    # 4. Setup Callbacks
    # -------------------------------
    callbacks = []

    # Logs batch-level info every 50 steps
    callbacks.append(ConsoleLogger(every_n_batches=100))

    # Checkpoints every epoch
    callbacks.append(CheckpointCallback(path="checkpoints-t/", every_n_epochs=1))

    # Evaluate validation dataset every epoch
    callbacks.append(EvaluationCallback(every_n_epochs=1))

    # Measure time per epoch and total training
    callbacks.append(TimeProfiler())

    # -------------------------------
    # 5. Create Engine
    # -------------------------------
    engine = Engine(
        task=task,
        callbacks=callbacks,
        # device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # -------------------------------
    # 6. Run Training
    # -------------------------------
    engine.train(epochs=5)

    logger.info("Training complete")

    # -------------------------------
    # 7. Optionally Run Evaluation Only
    # -------------------------------
    logger.info("Running final evaluation with validation set")
    engine.evaluate()

    logger.info("Job Complete")


if __name__ == "__main__":
    main()