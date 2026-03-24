from dataclasses import dataclass

from ml_runner.core.registries.callbacks import Callback
from ml_runner.core.engine.engine import EngineEvent


@dataclass
class EvaluationCallback:
    every_n_epochs: int = 1

    @Callback(EngineEvent.EPOCH_END)
    def on_epoch_end(self, engine, **kwargs):
        if engine.train_state.epoch % self.every_n_epochs == 0:
            engine.evaluate()
