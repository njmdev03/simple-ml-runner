from dataclasses import dataclass
from tqdm import tqdm
from ml_runner.core.log_utils import logger
from .base_callback import Callback


@dataclass
class EpochLogger(Callback):

    def on_epoch_start(self, engine):
        logger.info(f"Training starting on Epoch {engine.train_state.epoch}/{engine.train_state.epochs}")

        self.pbar = tqdm(
            total=engine.train_state.batches,
            desc=f"Epoch {engine.train_state.epoch}",
            leave=False,
            dynamic_ncols=True
        )

    def on_batch_end(self, engine):
        loss = engine.train_state.loss

        self.pbar.update(1)
        self.pbar.set_postfix(loss=f"{loss:.4f}")

    def on_epoch_end(self, engine):
        self.pbar.close()

        avg_loss = engine.train_state.total_loss / len(engine.task.train_loader)

        logger.info(f"Training on Epoch {engine.train_state.epoch}/{engine.train_state.epochs} finished")
        logger.info(f"train_loss={avg_loss:.4f}")