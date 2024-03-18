import glob
from cv2 import imread, resize, IMREAD_GRAYSCALE
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import numpy as np
from os import listdir


from typing import Union, List, Tuple, Dict


class LeafDatasetIllnessType(Dataset):
  def __init__(self, root: str, resize_size: Union[None, tuple, int]=None, transform: Union[None, object]=None):
    """
    Args:
      root (str): The root directory of the dataset.
      resize_size (Union[None|tuple|int]): The size to resize the images and masks to. If None, the images and masks will not be resized.
      transform (Union[None, object]): A transform to apply to the images and masks. If None, no transform will be applied.
      
    Returns:
        tuple: A tuple containing the image and class tensor.
    """
    super().__init__()
    self.root_dir = root
    self.root_dir += "/" if not self.root_dir.endswith("/") else ""
    self.transform = transform
    if isinstance(resize_size, int):
      self.resize: tuple = (resize_size, resize_size)
    self.resize: tuple = resize_size
    
    file_list = glob.glob(self.root_dir + "*/*.JPG")
    self.target_encode = [
      i.split('/')[-2].split('___')[1].lower().replace('_', ' ').replace('-', ' ').replace('(', '(').replace(')', ")")
      for i in file_list
    ]
    self.classes = sorted(list(set(self.target_encode)))
    self.class_to_idx = {label: idx for idx, label in enumerate(self.classes)} 
    self.targets = [self.class_to_idx[i] for i in self.target_encode]
    self.samples = list(zip(file_list, self.targets))
  
  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):
    img_path, class_id = self.samples[idx]
  
    # load image and mask
    img = imread(img_path)
    
    if self.resize:
      img = resize(img, self.resize)
      
    img_tensor = torch.from_numpy(img).float()
    img_tensor = img_tensor.permute(2, 0, 1)
    img_tensor = img_tensor / 255
    
    if self.transform:
      img_tensor = self.transform(img_tensor)
    
    return img_path, class_id

# %%