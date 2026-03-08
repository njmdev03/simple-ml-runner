from dataclasses import dataclass
from .base_callback import Callback

@dataclass
class EvaluationCallback(Callback):

    def on_epoch_end(self, engine):
        engine.evaluate()