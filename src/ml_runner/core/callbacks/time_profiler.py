import time
from ml_runner.core.log_utils import logger
from dataclasses import dataclass
from .base_callback import Callback

@dataclass
class TimeProfiler(Callback):

    def on_train_start(self, engine):
        self.train_start_time = time.time()

    def on_epoch_start(self, engine):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, engine):
        elapsed = time.time() - self.epoch_start_time

        logger.info(f"Epoch {engine.train_state.epoch} time {elapsed:.2f}s")

    def on_train_end(self, engine):
        total = time.time() - self.train_start_time

        logger.info(f"Total training time {total:.2f}s")