# %%
import torch
from torch import nn

class SequencialModel(nn.Module):
    def __init__(self, embedding_dim: int, vocab_size: int, hidden_size: int, num_layers: int, num_classes: int, padding_idx: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
    
# %%
if __name__ == '__main__':
    # embedding_dim: 10
    # 
    model = SequencialModel(10, 20, 3, 3, 0)
    print(model)
    x = torch.randint(0, 4, (5, 6))
    y = model(x)
    print(y.shape)
# %%