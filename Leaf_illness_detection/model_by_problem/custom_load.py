import glob
from cv2 import imread, resize
import torch
from torch.utils.data import Dataset
from os import listdir


from typing import Union


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
    
    self.initilize_sample()
    self.classes = sorted(list(set(self.illness_path.keys())))
    self.class_to_idx = {label: idx for idx, label in enumerate(self.classes)}
    
    samples, targets = [], []
    for k, v in self.illness_path.items():
      files = glob.glob(v)
      fl = len(files)
      target = [self.class_to_idx[k]]*fl
      samples.extend(list(zip(files, target)))
      targets.extend(target)
      
    self.samples = samples
    self.targets = targets
  
    
  def initilize_sample(self):
      folder_names = listdir(self.root_dir)
      illness_dict = {}
      for name in folder_names:
          parts = name.split('___')
        # plant illness
          illness_name = parts[1]

          illness_type = (
            parts[1]
            .lower()
            .replace('_', ' ')
            .replace('-', ' ')
            .replace('(', '(')
            .replace(')', ")")
          )
          
          if illness_type not in illness_dict:
              illness_dict[illness_type] = {f"[{illness_name[0].upper()}{illness_name[0].lower()}]{illness_name[1:]}"}
          else:
              illness_dict[illness_type].add(f"[{illness_name[0].upper()}{illness_name[0].lower()}]{illness_name[1:]}")
      # sort
      illness_dict = dict(sorted(illness_dict.items()))
      # check unique
      is_ones_illness = set([len(i) for i in illness_dict.values()]) == {1}

      if not is_ones_illness:
        assert f"The number of classes for each class is greater than 1. Please check the documentation. {set([len(i) for i in illness_dict.values()])}"

      illness_dict = {k: list(v)[0] for k, v in illness_dict.items()}
      # to path
      self.illness_path = {k: self.root_dir + "*" + v + "*/*.JPG" for k, v in illness_dict.items()}
      
      
  
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
    
    return img_tensor, class_id

# %%
    
class LeafDatasetPlantType(Dataset):
  def __init__(self, root: str, resize_size: Union[None, tuple, int]=None, transform: Union[None, object]=None):
    """
    Loads data for COVID-19 detection using semantic segmentation and classification.
    
    Args:
      root (str): The root directory of the dataset.
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
    
    self.initilize_sample()
    self.classes = sorted(list(set(self.plant_path.keys())))
    self.class_to_idx = {label: idx for idx, label in enumerate(self.classes)}
    
    samples, targets = [], []
    for k, v in self.plant_path.items():
      files = glob.glob(v)
      fl = len(files)
      target = [self.class_to_idx[k]]*fl
      samples.extend(list(zip(files, target)))
      targets.extend(target)
      
    self.samples = samples
    self.targets = targets
  
    
  def initilize_sample(self):
      folder_names = listdir(self.root_dir)
      plant_dict = {}
      for name in folder_names:
          parts = name.split('___')
        # plant type
          plant_name = parts[0]
          
          plant_class = (
              parts[0]
              .lower()
              .replace('_', ' ')
              .replace(',', '')
              .replace('(', '')
              .replace(')', '')
          )
          
          if plant_class not in plant_dict:
              plant_dict[plant_class] = {f"[{plant_name[0].upper()}{plant_name[0].lower()}]{plant_name[1:]}"}
          else:
              plant_dict[plant_class].add(f"[{plant_name[0].upper()}{plant_name[0].lower()}]{plant_name[1:]}")

      # check unique
      is_ones_plant = set([len(i) for i in plant_dict.values()]) == {1}

      if not is_ones_plant:
        assert f"The number of classes for each class is greater than 1. Please check the documentation. {set([len(i) for i in plant_dict.values()])}"

      plant_dict = {k: list(v)[0] for k, v in plant_dict.items()}
      # to path
      self.plant_path = {k: self.root_dir + "*" + v + "*/*.JPG" for k, v in plant_dict.items()}
  
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
    
    return img_tensor, class_id
  
# %%