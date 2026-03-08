from dataclasses import dataclass
from base_callback import Callback
from log_utils import logger

@dataclass
class BatchFileLogger(Callback):

    def on_batch_end(self, engine):
        logger.info(
            f"batch={engine.train_state.batch} "
            f"loss={engine.train_state.loss:.4f}"
        )