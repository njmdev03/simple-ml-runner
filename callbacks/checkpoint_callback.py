import torch
from pathlib import Path
from log_utils import logger
from dataclasses import dataclass
from .base_callback import Callback

@dataclass
class CheckpointCallback(Callback):

    # TODO: base_path, filename_template, metadata type
    path: str = "checkpoints/"

    def __post_init__(self):
        Path(self.path).mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, engine):
        filename = f"{self.path}/epoch_{engine.train_state.epoch}.pt"

        # TODO: Alternative metadata formats (json, yml, pth/pt file)
        # TODO: optionally save model weights/path, epoch, loss, metrics
        torch.save(
            engine.model.state_dict(),
            filename
        )

        logger.info(f"Saved checkpoint {filename}")