from log_utils import logger
from dataclasses import dataclass
from .base_callback import Callback

@dataclass
class ConsoleLogger(Callback):

    def on_batch_end(self, engine):
        loss = engine.train_state.loss.item()

        logger.info(f"batch={engine.train_state.batch} loss={loss}")