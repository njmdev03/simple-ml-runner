from dataclasses import dataclass

from ml_runner.core.log_utils import logger
from ml_runner.core.registries.callbacks import Callback
from ml_runner.core.engine.engine import EngineEvent


@dataclass
class BatchFileLogger:

    @Callback(EngineEvent.BATCH_END)
    def on_batch_end(self, engine):
        logger.info(
            f"batch={engine.train_state.batch} "
            f"loss={engine.train_state.loss:.4f}"
        )
