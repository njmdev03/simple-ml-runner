from dataclasses import dataclass

from ml_runner.core.log_utils import logger
from ml_runner.core.registries.callbacks import Callback
from ml_runner.core.engine.engine import EngineEvent


@dataclass
class BatchLogger:

    @Callback(EngineEvent.BATCH_END)
    def on_batch_end(self, engine):
        loss = engine.train_state.loss.item()
        logger.verbose(f"batch={engine.train_state.batch} loss={loss}")

    @Callback(EngineEvent.EVAL_BATCH_END)
    def on_eval_batch_end(self, engine):
        loss = engine.eval_state.loss.item()
        logger.verbose(f"batch={engine.eval_state.batch} loss={loss}")
