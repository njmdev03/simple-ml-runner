import torch
import torch.nn as nn
from registries.model_registry import ModelRegistry

@ModelRegistry.register("RNN")
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, rnn_type='GRU', dropout=0.0):
        """
        rnn_type: 'GRU', 'LSTM', or 'RNN'
        """
        super().__init__()
        self.rnn_type = rnn_type.upper()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        if self.rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        elif self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        elif self.rnn_type == 'RNN':
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        else:
            raise ValueError(f"Unknown rnn_type: {rnn_type}")

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        if self.rnn_type == 'LSTM':
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
            out, _ = self.rnn(x, (h0, c0))
        else:  # GRU or vanilla RNN
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
            out, _ = self.rnn(x, h0)

        # Use the last time step output for classification/regression
        out = self.fc(out[:, -1, :])
        return out
