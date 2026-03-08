from dataclasses import dataclass
from tqdm import tqdm
from log_utils import logger
from .base_callback import Callback


@dataclass
class EvalLogger(Callback):

    def on_eval_start(self, engine):
        logger.info(f"Started model evaluation")

        self.eval_pbar = tqdm(
            total=len(engine.task.val_loader),
            desc=f"Evaluation",
            leave=False,
            dynamic_ncols=True
        )

    def on_eval_batch_end(self, engine):
        loss = engine.eval_state.loss

        self.eval_pbar.update(1)
        self.eval_pbar.set_postfix(loss=f"{loss:.4f}")

    def on_eval_end(self, engine):
        self.eval_pbar.close()

        avg_loss = engine.eval_state.total_loss / engine.eval_state.batch
        metrics = engine.eval_state.metrics

        msg = f"val_loss={avg_loss:.4f}"

        for k, v in metrics.items():
            msg += f" {k}={v:.4f}"

        logger.info(f"Evaluation finished")
        logger.info(f"{msg}")