import torch
from torch import nn

class SequencialModel(nn.Module):
    def __init__(self, embedding_dim, hidden_size, hidden_lstm, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(embedding_dim, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_lstm, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out