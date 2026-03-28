import nltk
from nltk.tokenize import word_tokenize
from ml_runner.extensions.nlp.registries import Tokenizer

# Ensure tokenizer is downloaded
for resource in ['tokenizers/punkt', 'tokenizers/punkt_tab']:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource.split('/')[-1])

@Tokenizer("basic")
def basic_tokenizer(text):
    return word_tokenize(text.lower())

@Tokenizer("whitespace")
def whitespace_tokenizer(text):
    return text.split()
