import torch
from ml_runner.core.engine.engine import Engine
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
    from ml_runner.core.engine.event_manager import EventManager
    from ml_runner.core.registries.callbacks import Callback, attach
    task = MockTask()

    class MockCallback:
        def __init__(self):
            self.on_train_start = MagicMock()
            self.epoch_start = MagicMock()
            self.batch_start = MagicMock()
            self.batch_end = MagicMock()
            self.epoch_end = MagicMock()
            self.on_train_end = MagicMock()

        @Callback("on_train_start")
        def _on_train_start(self, engine, **kwargs): self.on_train_start(engine=engine, **kwargs)
        @Callback("epoch_start")
        def _epoch_start(self, engine, **kwargs): self.epoch_start(engine=engine, **kwargs)
        @Callback("batch_start")
        def _batch_start(self, engine, **kwargs): self.batch_start(engine=engine, **kwargs)
        @Callback("batch_end")
        def _batch_end(self, engine, **kwargs): self.batch_end(engine=engine, **kwargs)
        @Callback("epoch_end")
        def _epoch_end(self, engine, **kwargs): self.epoch_end(engine=engine, **kwargs)
        @Callback("on_train_end")
        def _on_train_end(self, engine, **kwargs): self.on_train_end(engine=engine, **kwargs)

    cb = MockCallback()
    em = EventManager()
    attach(cb, em)
    engine = Engine(task, event_manager=em)

    engine.train(epochs=1)

    # Check if model.train() was called
    task.model.train.assert_called()

    # Check if lifecycle events were triggered
    assert cb.on_train_start.called
    assert cb.epoch_start.called
    assert cb.batch_start.call_count == 2
    assert cb.batch_end.call_count == 2
    assert cb.epoch_end.called
    assert cb.on_train_end.called

def test_engine_evaluate_loop():
    from ml_runner.core.engine.event_manager import EventManager
    from ml_runner.core.registries.callbacks import Callback, attach
    task = MockTask()

    class MockEvalCallback:
        def __init__(self):
            self.on_eval_start = MagicMock()
            self.eval_batch_start = MagicMock()
            self.eval_batch_end = MagicMock()
            self.on_eval_end = MagicMock()

        @Callback("on_eval_start")
        def _on_eval_start(self, engine, **kwargs): self.on_eval_start(engine=engine, **kwargs)
        @Callback("eval_batch_start")
        def _eval_batch_start(self, engine, **kwargs): self.eval_batch_start(engine=engine, **kwargs)
        @Callback("eval_batch_end")
        def _eval_batch_end(self, engine, **kwargs): self.eval_batch_end(engine=engine, **kwargs)
        @Callback("on_eval_end")
        def _on_eval_end(self, engine, **kwargs): self.on_eval_end(engine=engine, **kwargs)

    cb = MockEvalCallback()
    em = EventManager()
    attach(cb, em)
    engine = Engine(task, event_manager=em)

    metrics_captured = {}
    def capture_metrics(engine, **kwargs):
        metrics_captured.update(engine.eval_state.total_metrics)

    cb.on_eval_end.side_effect = capture_metrics
    engine.evaluate()

    # Check if model.eval() was called
    task.model.eval.assert_called()

    # Check events
    assert cb.on_eval_start.called
    assert cb.eval_batch_start.call_count == 1
    assert cb.eval_batch_end.call_count == 1
    assert cb.on_eval_end.called
    assert metrics_captured["acc"] == 1.0
