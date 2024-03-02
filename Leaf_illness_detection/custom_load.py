import glob
from cv2 import imread
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from typing import Union


class LeafDataset(Dataset):
  def __init__(self, root: str, classes: dict, transform: Union[None, transforms]=None):
    
    self.root_dir = root
    self.transform = transform
    self.class_map = classes
    
    file_list = glob.glob(self.root_dir + "*/*[.JPG|.jpg|.jpeg]")
    self.data = [[i, i.split('/')[-2]] for i in file_list]
    
  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    img_path, class_name = self.data[idx]
    class_id = self.class_map[class_name]
    class_id = torch.tensor([class_id])
    
    img = imread(img_path)
    img_tensor = torch.from_numpy(img)
    img_tensor = img_tensor.permute(2, 0, 1)
    
    if self.transform:
      img = self.transform(img)
    
    return img_tensor, class_id