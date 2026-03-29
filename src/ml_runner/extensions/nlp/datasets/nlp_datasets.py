import torch
from torch.utils.data import Dataset
from ml_runner.core.registries import Dataset as RegistryDataset
from datasets import load_dataset
from ml_runner.extensions.nlp.utils.vocab import Vocab

from ml_runner.extensions.nlp.registries import TokenizerRegistry
_global_vocabs = {}

@RegistryDataset("WikiText2")
class WikiText2Dataset(Dataset):
    def __init__(self, train: bool = True, seq_len: int = 35, stride: int = None, tokenizer_name: str = "basic"):
        self.seq_len = seq_len
        self.stride = stride or seq_len
        split = "train" if train else "validation"
        cache_key = f"WikiText2_{tokenizer_name}_vocab"

        ds = load_dataset("wikitext", "wikitext-2-v1", split=split)

        tokenizer = TokenizerRegistry.get(tokenizer_name)

        tokens = []
        for line in ds:
            text = line['text'].strip()
            if text:
                tokens.extend(tokenizer(text))

        if cache_key not in _global_vocabs:
            v = Vocab(specials=['<UNK>', '<PAD>', '<SOS>', '<EOS>'])
            v.build_vocab([tokens])
            _global_vocabs[cache_key] = v
        self.vocab = _global_vocabs[cache_key]

        self.data = self.vocab.encode(tokens)
        # Calculate total number of samples based on stride
        self.num_samples = (len(self.data) - seq_len - 1) // self.stride + 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.seq_len
        x = self.data[start:end]
        y = self.data[start+1:end+1]

        if len(x) < self.seq_len:
            x = torch.cat([x, torch.tensor([self.vocab['<PAD>']] * (self.seq_len - len(x)))])
        if len(y) < self.seq_len:
            y = torch.cat([y, torch.tensor([self.vocab['<PAD>']] * (self.seq_len - len(y)))])

        return x, y

@RegistryDataset("Multi30k")
class Multi30kDataset(Dataset):
    def __init__(self, train: bool = True, max_len: int = 50, tokenizer_name: str = "basic"):
        self.max_len = max_len
        split = "train" if train else "validation"
        ds = load_dataset("bentrevett/multi30k", split=split)

        tokenizer = TokenizerRegistry.get(tokenizer_name)

        src_data_list = [tokenizer(item['en'].lower()) for item in ds]
        trg_data_list = [tokenizer(item['de'].lower()) for item in ds]

        cache_key_src = f"Multi30k_{tokenizer_name}_vocab_src"
        cache_key_trg = f"Multi30k_{tokenizer_name}_vocab_trg"

        if cache_key_src not in _global_vocabs:
            v = Vocab(specials=['<UNK>', '<PAD>', '<SOS>', '<EOS>'])
            v.build_vocab(src_data_list)
            _global_vocabs[cache_key_src] = v
        self.src_vocab = _global_vocabs[cache_key_src]

        if cache_key_trg not in _global_vocabs:
            v = Vocab(specials=['<UNK>', '<PAD>', '<SOS>', '<EOS>'])
            v.build_vocab(trg_data_list)
            _global_vocabs[cache_key_trg] = v
        self.trg_vocab = _global_vocabs[cache_key_trg]

        self.src_encoded = []
        self.trg_encoded = []

        pad_idx_src = self.src_vocab['<PAD>']
        pad_idx_trg = self.trg_vocab['<PAD>']

        for s, t in zip(src_data_list, trg_data_list):
            s_enc = self.src_vocab.encode(s[:max_len-2], add_sos=True, add_eos=True)
            t_enc = self.trg_vocab.encode(t[:max_len-2], add_sos=True, add_eos=True)

            if len(s_enc) < max_len:
                s_enc = torch.cat([s_enc, torch.tensor([pad_idx_src] * (max_len - len(s_enc)))])
            if len(t_enc) < max_len:
                t_enc = torch.cat([t_enc, torch.tensor([pad_idx_trg] * (max_len - len(t_enc)))])

            self.src_encoded.append(s_enc)
            self.trg_encoded.append(t_enc)

    def __len__(self):
        return len(self.src_encoded)

    def __getitem__(self, idx):
        return self.src_encoded[idx], self.trg_encoded[idx]
