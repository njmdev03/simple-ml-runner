import torch
from ml_runner.core.tasks.base_task import BaseTask
from ml_runner.core.registries import Task

from ml_runner.extensions.nlp.registries import EmbeddingRegistry
from ml_runner.core.registries.metric import MetricRegistry

@Task("text_generation")
class TextGenerationTask(BaseTask):
    def __init__(self, model, loss_fn, optimizer, train_loader, val_loader, device="cpu", metrics=None, **kwargs):
        vocab = getattr(train_loader.dataset, 'vocab', None)
        if vocab and hasattr(model, 'reconfigure_for_vocab'):
            embedding_type = getattr(model, 'embedding_type', "GensimLoader")
            embedding_name = kwargs.get('embedding_name') or getattr(model, 'embedding_name', None)

            if embedding_name:
                # Use registry to get the loader
                LoaderClass = EmbeddingRegistry.get(embedding_type)
                loader = LoaderClass(embedding_name)
                weights = loader.load(vocab)
                if weights is not None:
                    model.reconfigure_for_vocab(len(vocab), weights.shape[1], weights)
            else:
                emb_dim = getattr(model.embedding, 'embedding_dim', 0) if hasattr(model, 'embedding') and model.embedding else 0
                model.reconfigure_for_vocab(len(vocab), emb_dim)

        super().__init__(model, loss_fn, optimizer, train_loader, val_loader, device, metrics, **kwargs)

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
        # Prepare context for metrics that might need it (like BLEU needing vocab)
        context = {
            "vocab": getattr(self.val_loader.dataset, 'vocab', None),
            "loss_fn": self.loss_fn
        }

        for metric in self.metrics:
            name = MetricRegistry.get_name(metric)
            try:
                # Standardize metric calls to pass context
                results[name] = metric(outputs, targets, **context)
            except Exception:
                # Fallback to no context if the metric doesn't support it
                try:
                    results[name] = metric(outputs, targets)
                except Exception:
                    pass
        return results

@Task("machine_translation")
class MachineTranslationTask(BaseTask):
    def __init__(self, model, loss_fn, optimizer, train_loader, val_loader, device="cpu", metrics=None, **kwargs):
        src_vocab = getattr(train_loader.dataset, 'src_vocab', None)
        trg_vocab = getattr(train_loader.dataset, 'trg_vocab', None)

        if src_vocab and trg_vocab:
            embedding_type = getattr(model, 'embedding_type', "GensimLoader")
            embedding_name = kwargs.get('embedding_name') or getattr(model, 'embedding_name', None)

            if embedding_name:
                LoaderClass = EmbeddingRegistry.get(embedding_type)
                loader = LoaderClass(embedding_name)
                src_weights = loader.load(src_vocab)
                trg_weights = loader.load(trg_vocab)

                # Update model with both
                if hasattr(model, 'reconfigure_for_vocabs'):
                    model.reconfigure_for_vocabs(len(src_vocab), len(trg_vocab),
                                                src_weights.shape[1] if src_weights is not None else 0,
                                                src_weights, trg_weights)
            else:
                if hasattr(model, 'reconfigure_for_vocabs'):
                    emb_dim = getattr(model.encoder.embedding, 'embedding_dim', 0) if hasattr(model.encoder, 'embedding') and model.encoder.embedding else 0
                    model.reconfigure_for_vocabs(len(src_vocab), len(trg_vocab), emb_dim)

        super().__init__(model, loss_fn, optimizer, train_loader, val_loader, device, metrics, **kwargs)

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
        from ml_runner.core.registries.metric import MetricRegistry
        results = {}
        # Prepare context
        context = {
            "src_vocab": getattr(self.val_loader.dataset, 'src_vocab', None),
            "trg_vocab": getattr(self.val_loader.dataset, 'trg_vocab', None),
            "loss_fn": self.loss_fn
        }

        for metric in self.metrics:
            name = MetricRegistry.get_name(metric)
            try:
                results[name] = metric(outputs, targets, **context)
            except Exception:
                try:
                    results[name] = metric(outputs, targets)
                except Exception:
                    pass
        return results
