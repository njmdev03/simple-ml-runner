from dataclasses import dataclass
from typing import Optional

@dataclass
class Callback:
    every_n_batches: Optional[int] = None
    every_n_epochs: Optional[int] = None
    every_n_eval_batches: Optional[int] = None

    # FILTERS

    def _should_run_batch(self, engine):
        if self.every_n_batches is None:
            return True

        return engine.train_state.batch % self.every_n_batches == 0

    def _should_run_epoch(self, engine):
        if self.every_n_epochs is None:
            return True

        return engine.train_state.epoch % self.every_n_epochs == 0

    def _should_run_eval_batch(self, engine):
        if self.every_n_eval_batches is None:
            return True

        return engine.eval_state.batch % self.every_n_eval_batches == 0

    # EVENTS

    def on_train_start(self, engine): pass
    def on_train_end(self, engine): pass

    def on_eval_start(self, engine): pass
    def on_eval_end(self, engine): pass

    # FILTERING METHODS (Don't override these unless you know what you are doing)

    def epoch_start(self, engine):
        if self._should_run_epoch(engine):
            self.on_epoch_start(engine)

    def epoch_end(self, engine):
        if self._should_run_epoch(engine):
            self.on_epoch_end(engine)

    def batch_start(self, engine):
        if self._should_run_batch(engine):
            self.on_batch_start(engine)

    def batch_end(self, engine):
        if self._should_run_batch(engine):
            self.on_batch_end(engine)

    def eval_batch_start(self, engine):
        if self._should_run_eval_batch(engine):
            self.on_eval_batch_start(engine)

    def eval_batch_end(self, engine):
        if self._should_run_eval_batch(engine):
            self.on_eval_batch_end(engine)

    # FILTERED METHODS (override these)

    def on_epoch_start(self, engine): pass

    def on_epoch_end(self, engine): pass

    def on_batch_start(self, engine): pass

    def on_batch_end(self, engine): pass

    def on_eval_batch_start(self, engine): pass

    def on_eval_batch_end(self, engine): pass
