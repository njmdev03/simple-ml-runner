from ml_runner.core.registries.base import BaseRegistry

class TokenizerRegistry(BaseRegistry):
    _registry = {}

def Tokenizer(*names: str):
    return TokenizerRegistry.register(*names)

class EmbeddingRegistry(BaseRegistry):
    _registry = {}

def EmbeddingType(*names: str):
    return EmbeddingRegistry.register(*names)
