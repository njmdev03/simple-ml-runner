import torch
import os
import json
import logging

logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self, config, model: torch.nn.Module, device: torch.device, profiler=None):
        self.config = config
        self.model = model
        self.device = device
        self.profiler = profiler

        self.criteria = [(type(c).__name__, c) for c in self.config.TESTING_CRITERION if c is not None]

    def evaluate(self, loader, name="Test", loader_name=None):
        # Start a profiling segment if profiler is provided
        if self.profiler:
            p_key = f"eval_{name.lower().replace(' ', '_')}"

            if loader_name:
                p_key += f"_{loader_name.lower().replace(' ', '_')}"

            self.profiler.start(p_key)

        self.model.eval()

        eval_step = self.config.EVAL_STEP_FN

        # Initialize result containers
        criterion_losses = {crit_name: 0.0 for crit_name, _ in self.criteria}

        with torch.no_grad():
            for batch in loader:
                if eval_step is not None:
                    # Custom eval step: (model, batch, device) -> (outputs, targets, loss_dict | None)
                    res = eval_step(self.model, batch, self.device)
                    outputs, targets = res[0], res[1]
                    step_losses = res[2] if len(res) > 2 else None

                    # If custom step returns losses, record them
                    if step_losses:
                        for crit_name, loss_val in step_losses.items():
                            criterion_losses[crit_name] = criterion_losses.get(crit_name, 0.0) + loss_val

                    # Feed to metrics
                    for metric in self.config.METRICS.values():
                        metric.update(outputs, targets)
                else:
                    data, target = batch
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)

                    # Update Losses
                    for crit_name, criterion in self.criteria:
                        if hasattr(criterion, '__call__'):
                            loss = criterion(output, target)
                            criterion_losses[crit_name] += loss.item()

                    # Update Informational Metrics
                    for metric in self.config.METRICS.values():
                        metric.update(output, target)

        # Finalize results
        results = {}

        # 1. Losses (skip when using a custom eval step — loss is internal to the model)
        if self.config.EVAL_STEP_FN is None:
            for crit_name, total_loss in criterion_losses.items():
                avg_loss = total_loss / len(loader)
                results[f"{crit_name}_loss"] = avg_loss
                logger.info(f'{name} set: {crit_name} Average loss: {avg_loss:.4f}')

        # 2. Informational Metrics
        for m_name, metric in self.config.METRICS.items():
            val = metric.compute()
            metric.reset()
            # Some metrics (e.g. DetectionMAP) return a dict of sub-metrics
            if isinstance(val, dict):
                for subkey, subval in val.items():
                    results[subkey] = subval
                    logger.info(f'{name} set: {subkey}: {subval:.4f}')
            else:
                results[m_name] = val
                logger.info(f'{name} set: {m_name}: {val:.4f}')

        if self.profiler:
            duration = self.profiler.stop(p_key)
            logger.info(f"{name} set evaluation finished in {duration:.2f}s")

        return results

    # skip_keys is a set of (epoch, dataset_name) tuples
    def run_checkpoints(self, loader, loader_name = "", skip_keys=None):
        if skip_keys is None:
            skip_keys = set()

        cp_dir = self.config.CHECK_MODEL_DIR
        if not os.path.exists(cp_dir):
            return []

        checkpoints = [f for f in os.listdir(cp_dir) if f.endswith(".json")]
        all_results = []

        # Sort by epoch
        meta_list = []
        for cp in checkpoints:
            with open(os.path.join(cp_dir, cp), 'r') as f:
                meta_list.append(json.load(f))
        meta_list.sort(key=lambda x: x.get('epoch', 0))

        for meta in meta_list:
            epoch = meta.get('epoch', 0)
            if (epoch, loader_name) in skip_keys:
                continue

            cp_path = os.path.join(cp_dir, meta['checkpoint'])

            logger.info(f"--- Evaluating Checkpoint: Epoch {epoch} ---")

            self.model.load_state_dict(torch.load(cp_path, map_location=self.device))
            res = self.evaluate(loader, name=f"Epoch {epoch}", loader_name=loader_name)

            res['epoch'] = epoch
            res['source'] = meta['checkpoint']
            res['dataset'] = loader_name
            yield res
