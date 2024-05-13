import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import tv_tensors
import pandas as pd
import numpy as np


class FastrcnnDataset(Dataset):
  def __init__(self, root: str, subdir: str, transform: None|object=None):
    """

    """
    super().__init__()
    self.root_dir = root
    self.subdir = subdir
    self.transform = transform


    self.data = (
      pd.read_csv(self.root_dir + 'face_data.csv')
      .assign(
        file_path=lambda k: self.root_dir + '/' + self.subdir + '/' + k.gender + '/' + k.file_name,
        class_ = lambda k: np.where(k.gender == "Female faces", 1, 2)
      )
      [['file_path', 'x_min', 'y_min', 'x_max', 'y_max', 'class_']]
      .to_numpy()
    )



  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    img_path, x_min, y_min, x_max, y_max, class_ = self.data[idx]

    img = read_image(img_path)

    # image height and width
    a_h, a_w = img.shape[1:]

    # resize coordinates
    x_min = int(x_min * a_w)
    y_min = int(y_min * a_h)
    x_max = int(x_max * a_w)
    y_max = int(y_max * a_h)

    # define box
    box = torch.tensor([[x_min, y_min, x_max, y_max]])

    # define labels
    labels = torch.tensor([class_])

    # compute area of bounding box
    area = (x_max - x_min) * (y_max - y_min)

    # iscrowd
    iscrowd = torch.tensor([0])

    # create target tensor
    img = tv_tensors.Image(img)

    target = {}
    target["boxes"] = tv_tensors.BoundingBoxes([[x_min, y_min, x_max, y_max]], format="XYXY", canvas_size=(a_h, a_w))
    target["labels"] = labels
    target["image_id"] = idx
    target["area"] = torch.tensor([area])
    target["iscrowd"] = iscrowd

    if self.transform:
      img, target = self.transform(img, target)

    return img, target

