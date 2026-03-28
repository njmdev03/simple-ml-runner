import torch
import torch.nn as nn
from ml_runner.core.registries import Model

from ml_runner.extensions.nlp.utils.embeddings import get_pretrained_embeddings

@Model("RNN")
class RNNModel(nn.Module):
    def __init__(self, vocab_size: int = 1, embedding_dim: int = 0, hidden_dim: int = 128, num_layers: int = 1, pretrained_embeddings=None, embedding_type="GensimLoader", embedding_name=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding_type = embedding_type
        self.embedding_name = embedding_name
        self.reconfigure_for_vocab(vocab_size, embedding_dim, pretrained_embeddings)

    def reconfigure_for_vocab(self, vocab_size, embedding_dim, pretrained_embeddings=None):
        self.vocab_size = vocab_size
        self.use_one_hot = (embedding_dim == 0)

        if self.use_one_hot:
            rnn_input_dim = vocab_size
            self.embedding = None
        elif pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
            rnn_input_dim = self.embedding.embedding_dim
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            rnn_input_dim = embedding_dim

        # Only create if it doesn't exist or if dimensions changed
        if not hasattr(self, 'rnn') or self.rnn.input_size != rnn_input_dim:
            self.rnn = nn.RNN(rnn_input_dim, self.hidden_dim, self.num_layers, batch_first=True)
            self.fc = nn.Linear(self.hidden_dim, vocab_size)
        elif self.fc.out_features != vocab_size:
            # Vocab size changed but input dim didn't (e.g. embedding dim is same but vocab bigger)
            self.fc = nn.Linear(self.hidden_dim, vocab_size)

    def load_pretrained_embeddings(self, vocab, name):
        if not self.use_one_hot:
            weights = get_pretrained_embeddings(vocab, name)
            if weights is not None:
                self.reconfigure_for_vocab(len(vocab), weights.shape[1], weights)

    def forward(self, x, h=None):
        if self.use_one_hot:
            x = torch.nn.functional.one_hot(x, num_classes=self.vocab_size).float()
        else:
            x = self.embedding(x)
        out, h = self.rnn(x, h)
        out = self.fc(out)
        return out, h

@Model("GRU")
class GRUModel(nn.Module):
    def __init__(self, vocab_size: int = 1, embedding_dim: int = 0, hidden_dim: int = 128, num_layers: int = 1, pretrained_embeddings=None, embedding_type="GensimLoader", embedding_name=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding_type = embedding_type
        self.embedding_name = embedding_name
        self.reconfigure_for_vocab(vocab_size, embedding_dim, pretrained_embeddings)

    def reconfigure_for_vocab(self, vocab_size, embedding_dim, pretrained_embeddings=None):
        self.vocab_size = vocab_size
        self.use_one_hot = (embedding_dim == 0)

        if self.use_one_hot:
            rnn_input_dim = vocab_size
            self.embedding = None
        elif pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
            rnn_input_dim = self.embedding.embedding_dim
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            rnn_input_dim = embedding_dim

        if not hasattr(self, 'gru') or self.gru.input_size != rnn_input_dim:
            self.gru = nn.GRU(rnn_input_dim, self.hidden_dim, self.num_layers, batch_first=True)
            self.fc = nn.Linear(self.hidden_dim, vocab_size)
        elif self.fc.out_features != vocab_size:
            # Vocab size changed but input dim didn't (e.g. embedding dim is same but vocab bigger)
            self.fc = nn.Linear(self.hidden_dim, vocab_size)

    def load_pretrained_embeddings(self, vocab, name):
        if not self.use_one_hot:
            weights = get_pretrained_embeddings(vocab, name)
            if weights is not None:
                self.reconfigure_for_vocab(len(vocab), weights.shape[1], weights)

    def forward(self, x, h=None):
        if self.use_one_hot:
            x = torch.nn.functional.one_hot(x, num_classes=self.vocab_size).float()
        else:
            x = self.embedding(x)
        out, h = self.gru(x, h)
        out = self.fc(out)
        return out, h

@Model("LSTM")
class LSTMModel(nn.Module):
    def __init__(self, vocab_size: int = 1, embedding_dim: int = 0, hidden_dim: int = 128, num_layers: int = 1, pretrained_embeddings=None, embedding_type="GensimLoader", embedding_name=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding_type = embedding_type
        self.embedding_name = embedding_name
        self.reconfigure_for_vocab(vocab_size, embedding_dim, pretrained_embeddings)

    def reconfigure_for_vocab(self, vocab_size, embedding_dim, pretrained_embeddings=None):
        self.vocab_size = vocab_size
        self.use_one_hot = (embedding_dim == 0)

        if self.use_one_hot:
            rnn_input_dim = vocab_size
            self.embedding = None
        elif pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
            rnn_input_dim = self.embedding.embedding_dim
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            rnn_input_dim = embedding_dim

        if not hasattr(self, 'lstm') or self.lstm.input_size != rnn_input_dim:
            self.lstm = nn.LSTM(rnn_input_dim, self.hidden_dim, self.num_layers, batch_first=True)
            self.fc = nn.Linear(self.hidden_dim, vocab_size)
        elif self.fc.out_features != vocab_size:
            # Vocab size changed but input dim didn't (e.g. embedding dim is same but vocab bigger)
            self.fc = nn.Linear(self.hidden_dim, vocab_size)

    def load_pretrained_embeddings(self, vocab, name):
        if not self.use_one_hot:
            weights = get_pretrained_embeddings(vocab, name)
            if weights is not None:
                self.reconfigure_for_vocab(len(vocab), weights.shape[1], weights)

    def forward(self, x, hidden=None):
        if self.use_one_hot:
            x = torch.nn.functional.one_hot(x, num_classes=self.vocab_size).float()
        else:
            x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden

class Encoder(nn.Module):
    def __init__(self, vocab_size=1, emb_dim=0, hid_dim=128, n_layers=1, rnn_type="RNN"):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.rnn_type = rnn_type
        self.reconfigure_for_vocab(vocab_size, emb_dim)

    def reconfigure_for_vocab(self, vocab_size, emb_dim, weights=None):
        self.vocab_size = vocab_size
        self.use_one_hot = (emb_dim == 0)

        if self.use_one_hot:
            input_dim = vocab_size
            self.embedding = None
        elif weights is not None:
            self.embedding = nn.Embedding.from_pretrained(weights, freeze=False)
            input_dim = self.embedding.embedding_dim
        else:
            input_dim = emb_dim
            self.embedding = nn.Embedding(vocab_size, emb_dim)

        if not hasattr(self, 'rnn') or self.rnn.input_size != input_dim:
            if self.rnn_type == "LSTM":
                self.rnn = nn.LSTM(input_dim, self.hid_dim, self.n_layers, batch_first=True)
            elif self.rnn_type == "GRU":
                self.rnn = nn.GRU(input_dim, self.hid_dim, self.n_layers, batch_first=True)
            else:
                self.rnn = nn.RNN(input_dim, self.hid_dim, self.n_layers, batch_first=True)

    def forward(self, src):
        if self.use_one_hot:
            embedded = torch.nn.functional.one_hot(src, num_classes=self.vocab_size).float()
        else:
            embedded = self.embedding(src)
        outputs, hidden = self.rnn(embedded)
        return hidden

class Decoder(nn.Module):
    def __init__(self, vocab_size=1, emb_dim=0, hid_dim=128, n_layers=1, rnn_type="RNN"):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.rnn_type = rnn_type
        self.reconfigure_for_vocab(vocab_size, emb_dim)

    def reconfigure_for_vocab(self, vocab_size, emb_dim, weights=None):
        self.vocab_size = vocab_size
        self.use_one_hot = (emb_dim == 0)

        if self.use_one_hot:
            input_dim = vocab_size
            self.embedding = None
        elif weights is not None:
            self.embedding = nn.Embedding.from_pretrained(weights, freeze=False)
            input_dim = self.embedding.embedding_dim
        else:
            input_dim = emb_dim
            self.embedding = nn.Embedding(vocab_size, emb_dim)

        if not hasattr(self, 'rnn') or self.rnn.input_size != input_dim:
            if self.rnn_type == "LSTM":
                self.rnn = nn.LSTM(input_dim, self.hid_dim, self.n_layers, batch_first=True)
            elif self.rnn_type == "GRU":
                self.rnn = nn.GRU(input_dim, self.hid_dim, self.n_layers, batch_first=True)
            else:
                self.rnn = nn.RNN(input_dim, self.hid_dim, self.n_layers, batch_first=True)
            self.fc_out = nn.Linear(self.hid_dim, vocab_size)
        elif self.fc_out.out_features != vocab_size:
            self.fc_out = nn.Linear(self.hid_dim, vocab_size)

    def forward(self, input, hidden):
        input = input.unsqueeze(1) # [batch, 1]
        if self.use_one_hot:
            embedded = torch.nn.functional.one_hot(input, num_classes=self.vocab_size).float()
        else:
            embedded = self.embedding(input)
        output, hidden = self.rnn(embedded, hidden)
        prediction = self.fc_out(output) # [batch, 1, vocab_size]
        return prediction, hidden

@Model("RNN_Seq2Seq")
class RNNSeq2Seq(nn.Module):
    def __init__(self, src_vocab_size=1, trg_vocab_size=1, emb_dim=0, hid_dim=128, n_layers=1, embedding_type="GensimLoader", embedding_name=None):
        super().__init__()
        self.embedding_type = embedding_type
        self.embedding_name = embedding_name
        self.encoder = Encoder(src_vocab_size, emb_dim, hid_dim, n_layers, "RNN")
        self.decoder = Decoder(trg_vocab_size, emb_dim, hid_dim, n_layers, "RNN")

    def reconfigure_for_vocabs(self, src_vocab_size, trg_vocab_size, emb_dim, src_weights=None, trg_weights=None):
        self.encoder.reconfigure_for_vocab(src_vocab_size, emb_dim, src_weights)
        self.decoder.reconfigure_for_vocab(trg_vocab_size, emb_dim, trg_weights)

    def forward(self, src, trg):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        vocab_size = self.decoder.fc_out.out_features
        outputs = torch.zeros(batch_size, trg_len, vocab_size).to(src.device)
        hidden = self.encoder(src)

        input = trg[:, 0]
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden)
            outputs[:, t] = output.squeeze(1)
            input = trg[:, t]
        return outputs

@Model("GRU_Seq2Seq")
class GRUSeq2Seq(nn.Module):
    def __init__(self, src_vocab_size=1, trg_vocab_size=1, emb_dim=0, hid_dim=128, n_layers=1, embedding_type="GensimLoader", embedding_name=None):
        super().__init__()
        self.embedding_type = embedding_type
        self.embedding_name = embedding_name
        self.encoder = Encoder(src_vocab_size, emb_dim, hid_dim, n_layers, "GRU")
        self.decoder = Decoder(trg_vocab_size, emb_dim, hid_dim, n_layers, "GRU")

    def reconfigure_for_vocabs(self, src_vocab_size, trg_vocab_size, emb_dim, src_weights=None, trg_weights=None):
        self.encoder.reconfigure_for_vocab(src_vocab_size, emb_dim, src_weights)
        self.decoder.reconfigure_for_vocab(trg_vocab_size, emb_dim, trg_weights)

    def forward(self, src, trg):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        vocab_size = self.decoder.fc_out.out_features
        outputs = torch.zeros(batch_size, trg_len, vocab_size).to(src.device)
        hidden = self.encoder(src)
        input = trg[:, 0]
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden)
            outputs[:, t] = output.squeeze(1)
            input = trg[:, t]
        return outputs

@Model("LSTM_Seq2Seq")
class LSTMSeq2Seq(nn.Module):
    def __init__(self, src_vocab_size=1, trg_vocab_size=1, emb_dim=0, hid_dim=128, n_layers=1, embedding_type="GensimLoader", embedding_name=None):
        super().__init__()
        self.embedding_type = embedding_type
        self.embedding_name = embedding_name
        self.encoder = Encoder(src_vocab_size, emb_dim, hid_dim, n_layers, "LSTM")
        self.decoder = Decoder(trg_vocab_size, emb_dim, hid_dim, n_layers, "LSTM")

    def reconfigure_for_vocabs(self, src_vocab_size, trg_vocab_size, emb_dim, src_weights=None, trg_weights=None):
        self.encoder.reconfigure_for_vocab(src_vocab_size, emb_dim, src_weights)
        self.decoder.reconfigure_for_vocab(trg_vocab_size, emb_dim, trg_weights)

    def forward(self, src, trg):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        vocab_size = self.decoder.fc_out.out_features
        outputs = torch.zeros(batch_size, trg_len, vocab_size).to(src.device)
        hidden = self.encoder(src)
        input = trg[:, 0]
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden)
            outputs[:, t] = output.squeeze(1)
            input = trg[:, t]
        return outputs
