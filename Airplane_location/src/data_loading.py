from cv2 import imread, resize
import torch
from torch.utils.data import Dataset
import pandas as pd

from typing import Union


class YoloDataset(Dataset):
  def __init__(self, root: str, transform: Union[None, object]=None, resize_size=int|tuple|list, S: int=...):
    """

    """
    super().__init__()
    self.root_dir = root
    self.transform = transform
    if isinstance(resize_size, int):
      self.resize_size = (resize_size, resize_size)
    elif isinstance(resize_size, tuple) or isinstance(resize_size, list):
      if len(resize_size) != 2:
        raise ValueError("resize_size must be a tuple or list of length 2")
      self.resize_size = resize_size
    else:
      raise ValueError("resize_size must be an integer, tuple or list")
    self.S = S
    
    self.data: list = (
        pd.read_csv(
        self.root_dir + 'airplanes.csv', 
        header=None, 
        names=['image', 'x_min', 'y_min', 'x_max', 'y_max']
      )
      .assign(
        file_path=lambda k: self.root_dir + '/images/' + k.image,
      )
      [['file_path', 'x_min', 'y_min', 'x_max', 'y_max']]
      .to_numpy()
    )
  
    
    
  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    img_path, x_min, y_min, x_max, y_max = self.data[idx]
    
    # load image and mask
    img = imread(img_path)
    
    # actual height and width
    a_h, a_w = img.shape[:2] # height, width
    # resize height and width
    r_w, r_h = self.resize_size
    x_min = (x_min/a_w)
    y_min = (y_min/a_h)
    x_max = (x_max/a_w)
    y_max = (y_max/a_h)
    
    img = resize(img, self.resize_size)
    
    w, h, x, y = x_max - x_min, y_max - y_min, x_min, y_min
    
    # center of bounding box
    x_c = x + w/2
    y_c = y + h/2
    
    # Find grid cell
    i = int(x_c*self.S)
    j = int(y_c*self.S)
    
    # find x, y, w, h relative to grid cell
    x_cell = x_c*self.S - i
    y_cell = y_c*self.S - j
    w_cell = w*self.S
    h_cell = h*self.S
    
    # create target tensor
    target = torch.zeros((self.S, self.S, 5))
    target[j, i, 0] = x_cell
    target[j, i, 1] = y_cell
    target[j, i, 2] = w_cell
    target[j, i, 3] = h_cell
    target[j, i, 4] = 1
    
    img_tensor = torch.from_numpy(img).float()
    img_tensor = img_tensor.permute(2, 0, 1)
    img_tensor = img_tensor / 255
    
    # if self.transform:
    #   img_tensor= self.transform(img_tensor)
      
    sample = {"image": img_tensor, "target": target}
    
    return sample
  

