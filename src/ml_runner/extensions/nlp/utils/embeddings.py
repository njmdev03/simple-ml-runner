import torch
import torch.nn as nn
import numpy as np
import gensim.downloader as api
import logging
from ml_runner.extensions.nlp.registries import EmbeddingType

logger = logging.getLogger(__name__)

@EmbeddingType("GensimLoader")
class GensimEmbeddingLoader:
    """
    A registry-based embedding loader for Gensim-supported vectors.
    """
    def __init__(self, model_name="glove-wiki-gigaword-50"):
        self.model_name = model_name
        self._vectors = None

    def load(self, vocab):
        """
        Load vectors aligned with the provided vocab.
        """
        try:
            if self._vectors is None:
                logger.info(f"Fetching pre-trained embeddings: {self.model_name}...")
                self._vectors = api.load(self.model_name)

            emb_dim = self._vectors.vector_size
            weights = np.zeros((len(vocab), emb_dim))
            found_count = 0

            for i, token in enumerate(vocab.itos):
                if token in self._vectors:
                    weights[i] = self._vectors[token]
                    found_count += 1
                elif token.lower() in self._vectors:
                    weights[i] = self._vectors[token.lower()]
                    found_count += 1
                else:
                    weights[i] = np.random.normal(scale=0.6, size=(emb_dim,))

            logger.info(f"Embedded {found_count}/{len(vocab)} tokens using {self.model_name}")
            return torch.from_numpy(weights).float()

        except Exception as e:
            logger.error(f"Failed to load embeddings '{self.model_name}': {e}")
            return None

def get_pretrained_embeddings(vocab, embedding_name="glove-wiki-gigaword-50"):
    """
    Legacy compatibility wrapper for get_pretrained_embeddings.
    """
    loader = GensimEmbeddingLoader(embedding_name)
    return loader.load(vocab)
