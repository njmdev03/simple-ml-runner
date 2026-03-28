import torch
from ml_runner.core.tasks.base_task import BaseTask
from ml_runner.core.registries import Task

@Task("text_generation")
class TextGenerationTask(BaseTask):
    def training_step(self, batch):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        outputs, _ = self.model(x)
        loss = self.loss_fn(outputs.view(-1, outputs.size(-1)), y.view(-1))
        return loss, outputs, y

    def evaluation_step(self, batch):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        with torch.no_grad():
            outputs, _ = self.model(x)
            loss = self.loss_fn(outputs.view(-1, outputs.size(-1)), y.view(-1))
        return loss, outputs, y

    def compute_metrics(self, outputs, targets):
        results = {}
        # Get vocab from dataset
        vocab = getattr(self.val_loader.dataset, 'vocab', None)

        for metric in self.metrics:
            if metric.__name__ == 'perplexity':
                results['perplexity'] = metric(outputs, targets, self.loss_fn)
            elif metric.__name__ == 'bleu' and vocab:
                # Decode indices to strings
                preds = torch.argmax(outputs, dim=-1)
                decoded_preds = [" ".join(vocab.decode(p.tolist())) for p in preds]
                decoded_targets = [" ".join(vocab.decode(t.tolist())) for t in targets]
                results['bleu'] = metric(decoded_preds, decoded_targets)
            else:
                # Try generic call
                try:
                    results[metric.__name__] = metric(outputs, targets)
                except Exception:
                    pass
        return results

@Task("machine_translation")
class MachineTranslationTask(BaseTask):
    def training_step(self, batch):
        src, trg = batch
        src, trg = src.to(self.device), trg.to(self.device)
        outputs = self.model(src, trg)
        loss = self.loss_fn(outputs.view(-1, outputs.size(-1)), trg.view(-1))
        return loss, outputs, trg

    def evaluation_step(self, batch):
        src, trg = batch
        src, trg = src.to(self.device), trg.to(self.device)
        with torch.no_grad():
            outputs = self.model(src, trg)
            loss = self.loss_fn(outputs.view(-1, outputs.size(-1)), trg.view(-1))
        return loss, outputs, trg

    def compute_metrics(self, outputs, targets):
        results = {}
        # Get target vocab from dataset
        trg_vocab = getattr(self.val_loader.dataset, 'trg_vocab', None)

        for metric in self.metrics:
            if metric.__name__ == 'perplexity':
                results['perplexity'] = metric(outputs, targets, self.loss_fn)
            elif metric.__name__ == 'bleu' and trg_vocab:
                preds = torch.argmax(outputs, dim=-1)
                # Filter out special tokens like <PAD>, <SOS>, <EOS> for BLEU calculation if desired
                # But for now, just decode
                decoded_preds = [" ".join(trg_vocab.decode(p.tolist())) for p in preds]
                decoded_targets = [" ".join(trg_vocab.decode(t.tolist())) for t in targets]
                results['bleu'] = metric(decoded_preds, decoded_targets)
            else:
                try:
                    results[metric.__name__] = metric(outputs, targets)
                except Exception:
                    pass
        return results
