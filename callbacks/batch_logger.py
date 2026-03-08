from log_utils import logger
from dataclasses import dataclass
from .base_callback import Callback

@dataclass
class BatchLogger(Callback):

    def on_batch_end(self, engine):
        loss = engine.train_state.loss.item()

        logger.verbose(f"batch={engine.train_state.batch} loss={loss}")

    def on_eval_batch_end(self, engine):
        loss = engine.eval_state.loss.item()

        logger.verbose(f"batch={engine.eval_state.batch} loss={loss}")