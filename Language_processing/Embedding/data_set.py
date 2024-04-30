import torch
from torch.utils.data import Dataset
import numpy as np

class DatasetReviews(Dataset):
  def __init__(self, X, Y, vocab_size):
    self.X = X
    self.Y = Y
    self.v_dim = vocab_size

  def __len__(self):
    return len(self.Y)

  def __getitem__(self, index):
    xx = torch.LongTensor(self.X[index])
    yy = self.Y[index]
    
    return {'X': xx, 'Y': yy}