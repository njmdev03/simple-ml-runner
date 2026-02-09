import torch
import os
import json
from string import Template
from typing import Dict, Any, Callable

class Trainer:
    def __init__(self, config: Dict[str, Any], model: torch.nn.Module, device: torch.device, profiler=None):
        self.config = config
        self.model = model
        self.device = device
        self.profiler = profiler

        self.optimizer_cls = self._get_optimizer()
        self.criterion = self._get_criterion()

    def _get_optimizer(self):
        opt_name = self.config.get('OPTIMIZER')
        if isinstance(opt_name, str):
            import torch.optim as optim
            return getattr(optim, opt_name)
        return opt_name # might be a factory/lambda

    def _get_criterion(self):
        crit = self.config.get('TRAIN_CRITERION')
        if isinstance(crit, str):
            import torch.nn as nn
            return getattr(nn, crit)()
        return crit

    def train_epoch(self, loader, optimizer, epoch):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

            if not self.config.get('SILENT'):
                if batch_idx % 10 == 0:
                    print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(loader.dataset)} '
                          f'({100. * batch_idx / len(loader):.0f}%)]\tLoss: {loss.item():.6f}')

        avg_loss = total_loss / len(loader)
        accuracy = correct / total
        return avg_loss, accuracy

    def save_checkpoint(self, epoch, avg_loss, accuracy):
        cp_dir = self.config.get('CHECK_MODEL_DIR')
        if not os.path.exists(cp_dir):
            os.makedirs(cp_dir, exist_ok=True)

        cp_name_template = Template(self.config.get('CHECK_MODEL_NAME'))
        cp_file_name = cp_name_template.substitute(epoch=epoch) + ".pt"
        cp_path = os.path.join(cp_dir, cp_file_name)

        torch.save(self.model.state_dict(), cp_path)

        if self.config.get('SAVE_METADATA'):
            meta_path = os.path.join(cp_dir, cp_file_name.replace(".pt", ".json"))
            metadata = {
                "checkpoint": cp_file_name,
                "epoch": epoch,
                "loss": avg_loss,
                "accuracy": accuracy
            }
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=4)

        return cp_path

    def run(self, train_loader, eval_callback: Callable = None):
        lr = self.config.get('LEARNING_RATE')
        # Handle the case where OPTIMIZER is a lambda like in the user's example
        if hasattr(self.optimizer_cls, '__call__') and not isinstance(self.optimizer_cls, type):
            optimizer = self.optimizer_cls(self.model.parameters(), lr)
        else:
            optimizer = self.optimizer_cls(self.model.parameters(), lr=lr)

        epochs = self.config.get('EPOCHS')
        start_epoch = 1

        # Resume logic
        resume = self.config.get('RESUME')
        if resume:
            if isinstance(resume, str) and os.path.isfile(resume):
                self.load_checkpoint(resume)
            else:
                # Handle automatic resume from latest checkpoint
                latest_cp = self._get_latest_checkpoint()
                if latest_cp:
                    start_epoch = self.load_checkpoint(latest_cp) + 1

        if self.profiler:
            self.profiler.start("training")

        for epoch in range(start_epoch, epochs + 1):
            if self.profiler:
                self.profiler.start(f"epoch_{epoch}")

            loss, acc = self.train_epoch(train_loader, optimizer, epoch)

            if self.profiler:
                epoch_duration = self.profiler.stop(f"epoch_{epoch}")
                # self.profiler.record_epoch(epoch, epoch_duration)
                if not self.config.get('SILENT'):
                    print(f"Epoch {epoch} finished in {epoch_duration:.2f}s")

            # Checkpoint
            rate = self.config.get('CHECK_RATE')
            if rate > 0 and epoch % rate == 0:
                self.save_checkpoint(epoch, loss, acc)

            # Early Halt
            halt_cond = self.config.get('EARLY_HALT_CONDITION')
            halt_thresh = self.config.get('EARLY_HALT_THRESHOLD')

            if halt_cond == 'Loss' and loss < halt_thresh:
                print(f"Early halting: Loss {loss} < threshold {halt_thresh}")
                break
            elif halt_cond == 'Accuracy' and acc > halt_thresh:
                print(f"Early halting: Accuracy {acc} > threshold {halt_thresh}")
                break

            # Eval while training
            if self.config.get('TEST_WHILE_TRAINING') and eval_callback:
                eval_callback(epoch)

        if self.profiler:
            train_duration = self.profiler.stop("training")
            if not self.config.get('SILENT'):
                print(f"Total training time: {train_duration:.2f}s")

        # Final save
        final_path = self.config.get('FINAL_OUTPUT_PATH')
        os.makedirs(os.path.dirname(final_path), exist_ok=True) if os.path.dirname(final_path) else None
        torch.save(self.model.state_dict(), final_path)
        print(f"Final model saved to {final_path}")

    def load_checkpoint(self, path):
        print(f"Resuming from {path}")
        self.model.load_state_dict(torch.load(path, map_location=self.device))

        # Check if metadata exists
        meta_path = path.replace(".pt", ".json")
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                return meta.get('epoch', 0)
        return 0

    def _get_latest_checkpoint(self):
        cp_dir = self.config.get('CHECK_MODEL_DIR')
        if not os.path.exists(cp_dir):
            return None
        checkpoints = [f for f in os.listdir(cp_dir) if f.endswith(".json")]
        if not checkpoints:
            return None

        latest_meta = None
        max_epoch = -1

        for cp in checkpoints:
            with open(os.path.join(cp_dir, cp), 'r') as f:
                meta = json.load(f)
                if meta.get('epoch', -1) > max_epoch:
                    max_epoch = meta['epoch']
                    latest_meta = meta

        if latest_meta:
            return os.path.join(cp_dir, latest_meta['checkpoint'])
        return None
