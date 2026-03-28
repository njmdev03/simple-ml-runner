import torch
import torch.nn as nn
from ml_runner.core.registries import Model

@Model("RNN")
class RNNModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_layers: int = 1, use_pretrained: bool = False, pretrained_embeddings=None):
        super().__init__()
        if use_pretrained and pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, h=None):
        x = self.embedding(x)
        out, h = self.rnn(x, h)
        out = self.fc(out)
        return out, h

@Model("GRU")
class GRUModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_layers: int = 1, use_pretrained: bool = False, pretrained_embeddings=None):
        super().__init__()
        if use_pretrained and pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, h=None):
        x = self.embedding(x)
        out, h = self.gru(x, h)
        out = self.fc(out)
        return out, h

@Model("LSTM")
class LSTMModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_layers: int = 1, use_pretrained: bool = False, pretrained_embeddings=None):
        super().__init__()
        if use_pretrained and pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers, rnn_type="RNN"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, batch_first=True)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, batch_first=True)
        else:
            self.rnn = nn.RNN(emb_dim, hid_dim, n_layers, batch_first=True)

    def forward(self, src):
        outputs, hidden = self.rnn(self.embedding(src))
        return hidden

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers, rnn_type="RNN"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, batch_first=True)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, batch_first=True)
        else:
            self.rnn = nn.RNN(emb_dim, hid_dim, n_layers, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, vocab_size)

    def forward(self, input, hidden):
        input = input.unsqueeze(1) # [batch, 1]
        embedded = self.embedding(input)
        output, hidden = self.rnn(embedded, hidden)
        prediction = self.fc_out(output) # [batch, 1, vocab_size]
        return prediction, hidden

@Model("RNN_Seq2Seq")
class RNNSeq2Seq(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, emb_dim, hid_dim, n_layers):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, emb_dim, hid_dim, n_layers, "RNN")
        self.decoder = Decoder(trg_vocab_size, emb_dim, hid_dim, n_layers, "RNN")

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
    def __init__(self, src_vocab_size, trg_vocab_size, emb_dim, hid_dim, n_layers):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, emb_dim, hid_dim, n_layers, "GRU")
        self.decoder = Decoder(trg_vocab_size, emb_dim, hid_dim, n_layers, "GRU")

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
    def __init__(self, src_vocab_size, trg_vocab_size, emb_dim, hid_dim, n_layers):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, emb_dim, hid_dim, n_layers, "LSTM")
        self.decoder = Decoder(trg_vocab_size, emb_dim, hid_dim, n_layers, "LSTM")

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
