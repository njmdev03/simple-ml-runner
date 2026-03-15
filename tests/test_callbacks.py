from callbacks.base_callback import Callback
from unittest.mock import MagicMock

def test_callback_filtering():
    engine = MagicMock()

    # Test every_n_batches
    cb = Callback(every_n_batches=2)
    engine.train_state.batch = 1
    assert cb._should_run_batch(engine) is False
    engine.train_state.batch = 2
    assert cb._should_run_batch(engine) is True

    # Test every_n_epochs
    cb = Callback(every_n_epochs=5)
    engine.train_state.epoch = 4
    assert cb._should_run_epoch(engine) is False
    engine.train_state.epoch = 5
    assert cb._should_run_epoch(engine) is True

    # Test every_n_eval_batches
    cb = Callback(every_n_eval_batches=3)
    engine.eval_state.batch = 2
    assert cb._should_run_eval_batch(engine) is False
    engine.eval_state.batch = 3
    assert cb._should_run_eval_batch(engine) is True

def test_callback_events_triggering():
    cb = Callback(every_n_epochs=2)
    cb.on_epoch_start = MagicMock()

    engine = MagicMock()

    # Epoch 1: Should NOT run
    engine.train_state.epoch = 1
    cb.epoch_start(engine)
    cb.on_epoch_start.assert_not_called()

    # Epoch 2: SHOULD run
    engine.train_state.epoch = 2
    cb.epoch_start(engine)
    cb.on_epoch_start.assert_called_once_with(engine)
