import torch
import os
import json
import logging
from string import Template
from typing import Callable

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, config, model: torch.nn.Module, device: torch.device, profiler=None):
        self.config = config
        self.model = model
        self.device = device
        self.profiler = profiler

        self.optimizer_cls = self._get_optimizer()
        self.criterion = self.config.TRAIN_CRITERION

    def _get_optimizer(self):
        opt_name = self.config.OPTIMIZER
        if isinstance(opt_name, str):
            import torch.optim as optim
            return getattr(optim, opt_name)
        return opt_name # might be a factory/lambda

    def train_epoch(self, loader, optimizer, epoch):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
            optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

            if batch_idx % 10 == 0:
                logger.info(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(loader.dataset)} '
                            f'({100. * batch_idx / len(loader):.0f}%)]\tLoss: {loss.item():.6f}')

        avg_loss = total_loss / len(loader)
        accuracy = correct / total
        return avg_loss, accuracy

    def save_checkpoint(self, epoch, avg_loss, accuracy):
        cp_dir = self.config.CHECK_MODEL_DIR
        if not os.path.exists(cp_dir):
            os.makedirs(cp_dir, exist_ok=True)

        cp_name_template = Template(self.config.CHECK_MODEL_NAME)
        cp_file_name = cp_name_template.substitute(epoch=epoch) + ".pt"
        cp_path = os.path.join(cp_dir, cp_file_name)

        torch.save(self.model.state_dict(), cp_path)

        if self.config.SAVE_METADATA:
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
        lr = self.config.LEARNING_RATE
        # Handle the case where OPTIMIZER is a lambda like in the user's example
        if hasattr(self.optimizer_cls, '__call__') and not isinstance(self.optimizer_cls, type):
            optimizer = self.optimizer_cls(self.model.parameters(), lr)
        else:
            optimizer = self.optimizer_cls(self.model.parameters(), lr=lr)

        epochs = self.config.EPOCHS
        start_epoch = 1

        # Resume logic
        resume = self.config.RESUME
        if resume:
            if isinstance(resume, str) and os.path.isfile(resume):
                latest_cp_path = resume
            else:
                # Handle automatic resume from latest checkpoint
                latest_cp_path = self._get_latest_checkpoint()

            if latest_cp_path:
                start_epoch = self.load_checkpoint(latest_cp_path)

        if self.profiler:
            self.profiler.resume("training")

        try:
            for epoch in range(start_epoch, epochs + 1):
                if self.profiler:
                    self.profiler.start(f"epoch_{epoch}")

                loss, acc = self.train_epoch(train_loader, optimizer, epoch)

                if self.profiler:
                    epoch_duration = self.profiler.stop(f"epoch_{epoch}")
                    # self.profiler.record_epoch(epoch, epoch_duration)
                    logger.info(f"Epoch {epoch} finished in {epoch_duration:.2f}s")

                # Checkpoint
                rate = self.config.CHECK_RATE
                if rate > 0 and epoch % rate == 0:
                    self.save_checkpoint(epoch, loss, acc)

                # Early Halt
                halt_cond = self.config.EARLY_HALT_CONDITION
                halt_thresh = self.config.EARLY_HALT_THRESHOLD

                if halt_cond and halt_cond.value == 'Loss' and loss < halt_thresh:
                    logger.info(f"Early halting: Loss {loss} < threshold {halt_thresh}")
                    break
                elif halt_cond and halt_cond.value == 'Accuracy' and acc > halt_thresh:
                    logger.info(f"Early halting: Accuracy {acc} > threshold {halt_thresh}")
                    break

                # Eval while training
                if self.config.TEST_WHILE_TRAINING and eval_callback:
                    eval_callback(epoch)
        except KeyboardInterrupt:
            logger.warning("\nTraining interrupted by user!")

            # Block to save recovery checkpoint, removed for now due to training issues from not having
            # data set resuming.
            # logger.warning("\nTraining interrupted by user! Saving recovery checkpoint...")
            # epoch = epoch if 'epoch' in locals() else start_epoch
            # # We save with a special flag in metadata
            # cp_path = self.save_checkpoint(epoch, 0.0, 0.0)

            # # Update metadata to mark as interrupted
            # meta_path = cp_path.replace(".pt", ".json")
            # if os.path.exists(meta_path):
            #     with open(meta_path, 'r') as f:
            #         meta = json.load(f)
            #     meta['interrupted'] = True
            #     with open(meta_path, 'w') as f:
            #         json.dump(meta, f, indent=4)

            raise

        if self.profiler:
            train_duration = self.profiler.pause("training")
            logger.info(f"Total training time: {train_duration:.2f}s")

        # Final save
        final_path = self.config.FINAL_OUTPUT_PATH
        try:
            os.makedirs(os.path.dirname(final_path), exist_ok=True) if os.path.dirname(final_path) else None
            torch.save(self.model.state_dict(), final_path)
            logger.info(f"Final model saved to {final_path}")
        except Exception as e:
            logger.error(f"Error saving final model to {final_path}: {e}")

    def load_checkpoint(self, path):
        logger.info(f"Resuming from {path}")
        self.model.load_state_dict(torch.load(path, map_location=self.device))

        # Check if metadata exists
        meta_path = path.replace(".pt", ".json")
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                return meta.get('epoch', 0) + (0 if meta.get('interrupted', False) else 1)
        return 0

    def _get_latest_checkpoint(self):
        cp_dir = self.config.CHECK_MODEL_DIR
        if not os.path.exists(cp_dir):
            return None
        checkpoints = [f for f in os.listdir(cp_dir) if f.endswith(".json")]
        if not checkpoints:
            return None

        latest_meta = None
        interrupted_meta = None

        for cp in checkpoints:
            with open(os.path.join(cp_dir, cp), 'r') as f:
                meta = json.load(f)

                # Check for interrupted flag first
                if meta.get('interrupted'):
                    print("Found interrupted checkpoint")
                    # If multiple interrupted (unlikely), take the highest epoch one
                    if not interrupted_meta or meta.get('epoch', -1) > interrupted_meta.get('epoch', -1):
                        interrupted_meta = meta

                if not latest_meta or meta.get('epoch', -1) > latest_meta.get('epoch', -1):
                    print(f"Found latest checkpoint {meta}")
                    latest_meta = meta

        # Prioritize interrupted checkpoint
        meta_to_use = interrupted_meta or latest_meta

        if meta_to_use:
            print(f"Using checkpoint {meta_to_use}")
            return os.path.join(cp_dir, meta_to_use['checkpoint'])
        return None
