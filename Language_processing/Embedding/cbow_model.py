import torch
from torch import nn

class CBOW(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, padd_idx: int):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=padd_idx)
        self.linear = nn.Linear(embedding_dim, vocab_size)

        
    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        embeds = torch.sum(embeds, dim=1)
        out = self.linear(embeds)

        return out