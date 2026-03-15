import torch
from engine.engine import Engine
from unittest.mock import MagicMock

class MockTask:
    def __init__(self):
        self.model = MagicMock()
        self.optimizer = MagicMock()
        self.train_loader = [([1], [1]), ([2], [2])] # 2 batches
        self.val_loader = [([3], [3])] # 1 batch

    def training_step(self, batch):
        return torch.tensor(0.5, requires_grad=True), [0.8], [1]

    def evaluation_step(self, batch):
        return torch.tensor(0.4), [0.7], [1]

    def compute_metrics(self, outputs, targets):
        return {"acc": 1.0}

def test_engine_train_loop():
    task = MockTask()
    cb = MagicMock()
    engine = Engine(task, callbacks=[cb])

    engine.train(epochs=1)

    # Check if model.train() was called
    task.model.train.assert_called()

    # Check if lifecycle events were triggered
    # on_train_start, epoch_start, batch_start (x2), batch_end (x2), epoch_end, on_train_end
    assert cb.on_train_start.called
    assert cb.epoch_start.called
    assert cb.batch_start.call_count == 2
    assert cb.batch_end.call_count == 2
    assert cb.epoch_end.called
    assert cb.on_train_end.called

def test_engine_evaluate_loop():
    task = MockTask()
    cb = MagicMock()
    engine = Engine(task, callbacks=[cb])

    engine.evaluate()

    # Check if model.eval() was called
    task.model.eval.assert_called()

    # Check events
    assert cb.on_eval_start.called
    assert cb.eval_batch_start.call_count == 1
    assert cb.eval_batch_end.call_count == 1
    assert cb.on_eval_end.called

    # Check metrics in state
    # After evaluation finishes, eval_state is reset to a new EvalState()
    # So we should check it DURING the loop or verify it was updated.
    # Actually Engine.evaluate() resets it at the end.
    # To check the values, we can use a callback.

    metrics_captured = {}
    def capture_metrics(eng):
        metrics_captured.update(eng.eval_state.total_metrics)

    cb.on_eval_end.side_effect = capture_metrics
    engine.evaluate()
    assert metrics_captured["acc"] == 1.0
