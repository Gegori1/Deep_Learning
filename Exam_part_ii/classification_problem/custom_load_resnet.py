import glob
from cv2 import imread, resize, IMREAD_GRAYSCALE
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import numpy as np


from typing import Union, List, Tuple, Dict


class LungDatasetClassif(Dataset):
  def __init__(self, root: str, resize_size: Union[None, tuple, int]=None, transform: Union[None, object]=None):
    """
    Loads data for COVID-19 detection using semantic segmentation and classification.
    
    Args:
      root (str): The root directory of the dataset.
      classes (dict): A dictionary that maps class names to integers.
      resize_size (Union[None|tuple|int]): The size to resize the images and masks to. If None, the images and masks will not be resized.
      transform (Union[None, object]): A transform to apply to the images and masks. If None, no transform will be applied.
      
    Returns:
        tuple: A tuple containing the image and class tensor.
    """
    super().__init__()
    self.root_dir = root
    self.transform = transform
    if isinstance(resize_size, int):
      self.resize: tuple = (resize_size, resize_size)
    self.resize: tuple = resize_size
    
    file_list: List[str] = glob.glob(self.root_dir + '*/images/*.png')
    self.target_encode: List[str]  = [i.split('/')[-3] for i in file_list]
    self.classes: List[str] = sorted(list(set(self.target_encode)))
    self.class_to_idx: Dict[str, int] = {label: idx for idx, label in enumerate(self.classes)} 
    self.targets: List[int] = [self.class_to_idx[i] for i in self.target_encode]
    self.samples: List[Tuple[str, int]] = list(zip(file_list, self.targets))
  
    
    
  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):
    img_path, class_id = self.samples[idx]
    msk_path = img_path.replace("images", "lung masks")
  
    # load image and mask
    img = imread(img_path, flags=IMREAD_GRAYSCALE)
    msk = imread(msk_path, flags=IMREAD_GRAYSCALE)

    
    if self.resize:
      img = resize(img, self.resize)
      msk = resize(msk, self.resize)
      
    msk = np.expand_dims(msk, axis=0)
    img = np.expand_dims(img, axis=0)
      
    img_tensor = torch.from_numpy(img).float()
    # img_tensor = img_tensor.permute(2, 0, 1)
    img_tensor = img_tensor / 255
    
    msk_tensor = torch.from_numpy(msk).float()
    # img_tensor = img_tensor.permute(2, 0, 1)
    msk_tensor = msk_tensor / 255
    
    if self.transform and (not isinstance(self.transform, transforms.Compose)):
      img_tensor, msk_tensor = self.transform(img_tensor, msk_tensor)
    elif self.transform:
      img_tensor = self.transform(img_tensor)
      
    samples = {'image': img_tensor, 'mask': msk_tensor, 'class': class_id}
    
    return samples

# %%