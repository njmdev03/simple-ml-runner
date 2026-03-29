import torch
import math
from ml_runner.core.registries import Metric

@Metric("perplexity")
def perplexity(outputs, targets, **kwargs):
    """
    Perplexity calculation: exp(loss)
    """
    loss_fn = kwargs.get('loss_fn', torch.nn.CrossEntropyLoss())

    with torch.no_grad():
        loss = loss_fn(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        return math.exp(loss.item())


_bleu_scorer = None

@Metric("bleu")
def bleu_metric(outputs, targets, **kwargs):
    """
    BLEU score calculation. Requires 'vocab' or 'trg_vocab' in kwargs.
    """
    global _bleu_scorer
    if _bleu_scorer is None:
        try:
            from sacrebleu.metrics import BLEU
            _bleu_scorer = BLEU()
        except ImportError:
            return 0.0

    vocab = kwargs.get('vocab') or kwargs.get('trg_vocab')
    if not vocab:
        return 0.0

    with torch.no_grad():
        preds = torch.argmax(outputs, dim=-1)

        # Decode indices to strings
        decoded_preds = [" ".join(vocab.decode(p.tolist())) for p in preds]
        decoded_targets = [" ".join(vocab.decode(t.tolist())) for t in targets]

        score = _bleu_scorer.corpus_score(decoded_preds, [decoded_targets])
        return score.score
