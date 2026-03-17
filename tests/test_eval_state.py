from ml_runner.core.engine.eval_state import EvalState

def test_eval_state_sum_metrics():
    state = EvalState()

    metrics1 = {"accuracy": 0.8, "loss": 0.5}
    state.sum_metrics(metrics1)
    assert state.total_metrics["accuracy"] == 0.8
    assert state.total_metrics["loss"] == 0.5

    metrics2 = {"accuracy": 0.2, "loss": 0.3}
    state.sum_metrics(metrics2)
    assert state.total_metrics["accuracy"] == 1.0
    assert state.total_metrics["loss"] == 0.8

def test_eval_state_safe_sum():
    state = EvalState()

    # Both present
    assert state._safe_sum({"a": 1}, {"a": 2}, "a") == 3

    # One present
    assert state._safe_sum({"a": 1}, {}, "a") == 1
    assert state._safe_sum({}, {"a": 2}, "a") == 2

    # None present
    assert state._safe_sum({}, {}, "a") is None
