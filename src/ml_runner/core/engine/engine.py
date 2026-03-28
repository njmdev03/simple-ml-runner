import torch
from enum import Enum
from typing import Union

from ml_runner.core.engine.train_state import TrainState
from ml_runner.core.engine.eval_state import EvalState
from ml_runner.core.engine.event_manager import EventManager


class EngineEvent(Enum):
    # Job lifecycle
    JOB_START = "on_job_start"
    JOB_END = "on_job_end"

    # Training lifecycle
    TRAIN_START = "on_train_start"
    TRAIN_END = "on_train_end"

    EPOCH_START = "epoch_start"
    EPOCH_END = "epoch_end"

    BATCH_START = "batch_start"
    BATCH_END = "batch_end"

    # Evaluation lifecycle
    EVAL_START = "on_eval_start"
    EVAL_END = "on_eval_end"

    EVAL_BATCH_START = "eval_batch_start"
    EVAL_BATCH_END = "eval_batch_end"


class Engine:

    def __init__(
        self,
        task,
        event_manager=None,
    ):

        self.task = task
        self.model = task.model

        self.train_state = TrainState()
        self.eval_state = EvalState()

        # Use provided EventManager or create a new empty one
        self.event_manager = event_manager or EventManager()

    def _trigger(self, event: Union[EngineEvent, str], **kwargs):
        """
        Dispatches an event over the event bus.
        The current engine instance is automatically added to kwargs.
        """
        self.event_manager.emit(event, engine=self, **kwargs)

    def trigger(self, event, **kwargs):
        """
        Dispatches an event over the event bus.
        """
        self.event_manager.emit(event, **kwargs)

    def train(self, epochs):
        self.train_state.epochs = epochs

        self._trigger(EngineEvent.TRAIN_START)

        for epoch in range(1, epochs + 1):
            self.model.train()
            self.train_state.epoch = epoch
            self.train_state.batches = len(self.task.train_loader)

            self._trigger(EngineEvent.EPOCH_START)

            for batch in self.task.train_loader:

                self.train_state.batch += 1

                self._trigger(EngineEvent.BATCH_START)

                loss, outputs, targets = self.task.training_step(batch)

                self.task.optimizer.zero_grad()
                loss.backward()
                self.task.optimizer.step()

                self.train_state.loss = loss.detach()

                if self.train_state.total_loss is not None:
                    self.train_state.total_loss = self.train_state.total_loss + loss.detach()
                else:
                    self.train_state.total_loss = loss.detach()

                self.train_state.outputs = outputs
                self.train_state.targets = targets

                self._trigger(EngineEvent.BATCH_END)

            self._trigger(EngineEvent.EPOCH_END)

            # Reset training state, keeping total epochs count
            epochs_t = self.train_state.epochs

            self.train_state = TrainState()

            self.train_state.epochs = epochs_t

        self._trigger(EngineEvent.TRAIN_END)

        self.train_state = TrainState()

    def evaluate(self):
        self.eval_state.epoch = self.train_state.epoch
        self.eval_state.batches = len(self.task.val_loader)

        self._trigger(EngineEvent.EVAL_START)

        self.model.eval()

        with torch.no_grad():
            for batch in self.task.val_loader:
                self.eval_state.batch += 1

                self._trigger(EngineEvent.EVAL_BATCH_START)

                loss, outputs, targets = self.task.evaluation_step(batch)

                self.eval_state.metrics = self.task.compute_metrics(outputs, targets)
                self.eval_state.sum_metrics(self.eval_state.metrics)

                self.eval_state.loss = loss.detach()

                if self.eval_state.total_loss is not None:
                    self.eval_state.total_loss = self.eval_state.total_loss + loss.detach()
                else:
                    self.eval_state.total_loss = loss.detach()

                self.eval_state.outputs = outputs
                self.eval_state.targets = targets

                self._trigger(EngineEvent.EVAL_BATCH_END)

        self._trigger(EngineEvent.EVAL_END)

        self.eval_state = EvalState()
