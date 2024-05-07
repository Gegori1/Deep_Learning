from cv2 import imread, resize, cvtColor, COLOR_BGR2RGB
import albumentations as A
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class YoloDataset(Dataset):
  def __init__(self, root: str, subdir: str, transform: None|object=None, resize_size=int|tuple|list, S: int=...):
    """

    """
    super().__init__()
    self.root_dir = root
    self.transform = transform
    self.subdir = subdir
    if isinstance(resize_size, int):
      self.resize_size = (resize_size, resize_size)
    elif isinstance(resize_size, tuple) or isinstance(resize_size, list):
      if len(resize_size) != 2:
        raise ValueError("resize_size must be a tuple or list of length 2")
      self.resize_size = resize_size
    else:
      raise ValueError("resize_size must be an integer, tuple or list")
    self.S = S
    
    self.data = (
      pd.read_csv(self.root_dir + 'face_data.csv')
      .assign(
        file_path=lambda k: self.root_dir + '/' + self.subdir + '/' + k.gender + '/' + k.file_name,
        class_ = lambda k: np.where(k.gender == "Female faces", 1, 0)
      )
      [['file_path', 'x_min', 'y_min', 'x_max', 'y_max', 'class_']]
      .to_numpy()
    )
  
    
    
  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    img_path, x_min, y_min, x_max, y_max, class_ = self.data[idx]
    
    # load image and mask
    img = imread(img_path)
    img = cvtColor(img, COLOR_BGR2RGB)
    
    # actual height and width
    a_h, a_w = img.shape[:2] # height, width
    # resize height and width
    r_w, r_h = self.resize_size
    # x_min = (x_min/a_w)
    # y_min = (y_min/a_h)
    # x_max = (x_max/a_w)
    # y_max = (y_max/a_h)
    
    img = resize(img, self.resize_size)
    
    x_max = min(x_max, 1)
    x_min = max(x_min, 0)
    y_max = min(y_max, 1)
    y_min = max(y_min, 0)
    
    w, h, x, y = x_max - x_min, y_max - y_min, x_min, y_min
    
    # center of bounding box
    x_c = x + w/2
    y_c = y + h/2
    
    # transform if available
    if self.transform and isinstance(self.transform, A.core.composition.Compose):
      tranformed = self.transform(image=img, bboxes=[[x_c, y_c, w, h, class_]])
      
      img = tranformed["image"]
      w, h, x, y, *_ = tranformed["bboxes"][0]
    
    elif self.transform:
      raise ValueError("transform must be an instance of albumentations.Compose")
    
    # Find grid cell
    i = int(x_c*self.S)
    j = int(y_c*self.S)
    
    # create target tensor
    target = torch.zeros((self.S, self.S, 6))
    target[j, i, 0] = x_c
    target[j, i, 1] = y_c
    target[j, i, 2] = w
    target[j, i, 3] = h
    target[j, i, 4] = 1
    target[j, i, 5] = 1 - class_
    target[j, i, 6] = class_
    
    img_tensor = torch.from_numpy(img).float()
    img_tensor = img_tensor.permute(2, 0, 1)
    img_tensor = img_tensor / 255
      
    sample = {"image": img_tensor, "target": target}
    
    return sample
  
if __name__ == "__main__":
  data_path = "../Archive_joint"
