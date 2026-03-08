from .state import State
from dataclasses import dataclass

@dataclass
class TrainState(State):
    # Current epoch
    epoch: int = 0
    # Total epochs to run
    epochs: int = 0