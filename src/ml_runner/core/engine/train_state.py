from dataclasses import dataclass

from ml_runner.core.engine.state import State


@dataclass
class TrainState(State):
    # Total epochs to run
    epochs: int = 0
