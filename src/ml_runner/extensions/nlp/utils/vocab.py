from collections import Counter
import torch

class Vocab:
    def __init__(self, min_freq=1, specials=None):
        self.itos = specials or []
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        self.min_freq = min_freq

    def build_vocab(self, data):
        counter = Counter()
        for tokens in data:
            counter.update(tokens)
        
        for token, freq in counter.most_common():
            if freq >= self.min_freq:
                if token not in self.stoi:
                    self.stoi[token] = len(self.itos)
                    self.itos.append(token)

    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi.get('<UNK>', 0))

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens, add_sos=False, add_eos=False):
        indices = [self[t] for t in tokens]
        if add_sos:
            indices = [self['<SOS>']] + indices
        if add_eos:
            indices = indices + [self['<EOS>']]
        return torch.tensor(indices, dtype=torch.long)

    def decode(self, indices):
        return [self.itos[i] for i in indices]
