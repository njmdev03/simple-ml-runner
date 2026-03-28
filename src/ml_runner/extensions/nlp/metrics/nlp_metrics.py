import torch
import math
from ml_runner.core.registries import Metric
from sacrebleu.metrics import BLEU

@Metric("perplexity")
def perplexity(outputs, targets, loss_fn=None):
    """
    Perplexity calculation: exp(loss)
    Assumes outputs are logits and targets are indices.
    """
    if loss_fn is None:
        loss_fn = torch.nn.CrossEntropyLoss()

    loss = loss_fn(outputs.view(-1, outputs.size(-1)), targets.view(-1))
    return math.exp(loss.item())

@Metric("bleu")
def bleu_score(decoded_preds, decoded_targets):
    """
    Real BLEU score using sacrebleu or nltk.
    Expects lists of strings.
    """
    # Using sacrebleu for standardized BLEU
    bleu = BLEU()
    # sacrebleu expects a list of hypotheses and a list of lists of references
    # decoded_preds: [batch_size] strings
    # decoded_targets: [batch_size] strings
    score = bleu.corpus_score(decoded_preds, [decoded_targets])
    return score.score
