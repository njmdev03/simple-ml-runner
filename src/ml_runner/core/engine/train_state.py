from .state import State
from dataclasses import dataclass

@dataclass
class TrainState(State):
    # Total epochs to run
    epochs: int = 0