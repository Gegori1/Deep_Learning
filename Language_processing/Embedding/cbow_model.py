# %%
import torch
from torch import nn

class CBOW(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, padd_idx: int):
        super().__init__()
        self.relu = nn.ReLU()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=padd_idx)
        self.linear = nn.Linear(embedding_dim, vocab_size)

        
    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        embeds = torch.sum(embeds, dim=1)
        out = self.linear(embeds)
        out = self.relu(out)

        return out
    
# %%
if __name__ == '__main__':
    model = CBOW(6_000, 300, 0)
    print(model)
    x = torch.randint(0, 10, (5, 2))
    y = model(x)
    print(y.shape)
# %%
