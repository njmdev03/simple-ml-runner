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

        self.criteria = [(type(c).__name__, c) for c in self.config.TESTING_CRITERION]



    def evaluate(self, loader, name="Test", loader_name=None):
        # Start a profiling segment if profiler is provided
        if self.profiler:
            p_key = f"eval_{name.lower().replace(' ', '_')}"

            if loader_name:
                p_key += f"_{loader_name.lower().replace(' ', '_')}"

            self.profiler.start(p_key)

        self.model.eval()
        results = {}

        with torch.no_grad():
            for crit_name, criterion in self.criteria:
                total_loss = 0
                correct = 0
                total = 0

                for data, target in loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)

                    if hasattr(criterion, '__call__'):
                        loss = criterion(output, target)
                        total_loss += loss.item()

                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += target.size(0)

                avg_loss = total_loss / len(loader)
                accuracy = correct / total

                results[f"{crit_name}_loss"] = avg_loss
                results[f"{crit_name}_accuracy"] = accuracy

                logger.info(f'{name} set: {crit_name} Average loss: {avg_loss:.4f}, Accuracy: {correct}/{total} ({accuracy*100:.2f}%)')

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
