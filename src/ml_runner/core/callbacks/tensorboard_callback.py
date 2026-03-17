from dataclasses import dataclass
from .base_callback import Callback

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

@dataclass
class TensorboardCallback(Callback):
    """
    Logs metrics and timing to Tensorboard.
    """

    def __post_init__(self):
        self.writer = None

    def on_train_start(self, engine):
        if SummaryWriter and not self.writer:
            self.writer = SummaryWriter(log_dir=self.log_dir)

    def on_epoch_end(self, engine):
        if not self.writer: return
        epoch = engine.train_state.epoch

        if engine.train_state.batch > 0:
            loss = engine.train_state.total_loss / engine.train_state.batch
            self.writer.add_scalar("Train/Loss", loss, epoch)

        timing = engine.metadata.get("timing", {})
        if "epoch_duration" in timing:
            self.writer.add_scalar("Time/Epoch", timing["epoch_duration"], epoch)

    def on_eval_end(self, engine):
        if not self.writer: return
        epoch = engine.eval_state.epoch

        if engine.eval_state.batch > 0:
            loss = engine.eval_state.total_loss / engine.eval_state.batch
            self.writer.add_scalar("Eval/Loss", loss, epoch)
            for k, v in engine.eval_state.total_metrics.items():
                self.writer.add_scalar(f"Eval/{k}", v / engine.eval_state.batch, epoch)

    def on_train_end(self, engine):
        if self.writer:
            self.writer.close()
            self.writer = None
