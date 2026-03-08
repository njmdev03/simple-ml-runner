from typing import Optional, Any
from dataclasses import dataclass

@dataclass
class State:
    # Current batch number
    batch: int = 0
    # Total batches to run
    batches: int = 0

    # Current batch loss
    loss: Optional[Any] = None
    # Running total loss
    total_loss: Optional[Any] = None
    # Current batch outputs
    outputs: Optional[Any] = None
    # Current batch targets
    targets: Optional[Any] = None