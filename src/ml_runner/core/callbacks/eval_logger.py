from dataclasses import dataclass
from tqdm import tqdm

from ml_runner.core.log_utils import logger
from ml_runner.core.registries.callbacks import Callback
from ml_runner.core.engine.engine import EngineEvent


@dataclass
class EvalLogger:

    @Callback(EngineEvent.EVAL_START)
    def on_eval_start(self, engine, **kwargs):
        logger.info(f"Started model evaluation")

        self.eval_pbar = tqdm(
            total=len(engine.task.val_loader),
            desc=f"Evaluation",
            leave=False,
            dynamic_ncols=True
        )

    @Callback(EngineEvent.EVAL_BATCH_END)
    def on_eval_batch_end(self, engine, **kwargs):
        loss = engine.eval_state.loss

        self.eval_pbar.update(1)
        self.eval_pbar.set_postfix(loss=f"{loss:.4f}")

    @Callback(EngineEvent.EVAL_END)
    def on_eval_end(self, engine, **kwargs):
        self.eval_pbar.close()

        avg_loss = engine.eval_state.total_loss / engine.eval_state.batch
        metrics = engine.eval_state.metrics

        msg = f"val_loss={avg_loss:.4f}"

        for k, v in metrics.items():
            msg += f" {k}={v:.4f}"

        logger.info(f"Evaluation finished")
        logger.info(f"{msg}")
