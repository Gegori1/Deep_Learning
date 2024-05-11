import glob
from cv2 import BORDER_CONSTANT
import torch
# from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
import numpy as np
from dataset import YoloDataset
import albumentations as A

def YoloLoaderTrainEval(
    path_to_data: str,
    subdir: str,
    batch_size: int, 
    test_size: float = 0.2,
    resize: int|tuple|list = ...,
    grid: int = 7,
    workers: int = 4,
    pin_memory_device: None|object = None,
    random_seed: int = 42
  ):
  """
    Load data from the Dataset and batch it into a DataLoader.

    Args:
    path_to_data: str: Path to the directory containing subdirectories with images.
    batch_size: int: Batch size.
    test_size: float: Proportion of the dataset to include in the test split.
    resize: Union[None, int, tuple, list]: Resize images to this size.
    grid: int: Grid size. Same for each dimension.
    workers: int: Number of workers.
    pin_memory_device: Union[None, object]: Device to pin memory to.
    random_seed: int: Seed for the train-test split.
    
    Returns:
    train and validation data. (data_train, data_val)
  """
  
  tnsfm = A.Compose([
    A.PadIfNeeded(min_height=resize, min_width=resize, always_apply=True, border_mode=BORDER_CONSTANT, value=0),
    A.Rotate(limit=90, p=0.3),
    A.RandomSizedBBoxSafeCrop(width=int(0.8*resize), height=int(0.8*resize)),
    A.PadIfNeeded(min_height=resize, min_width=resize, always_apply=True, border_mode=BORDER_CONSTANT, value=0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
  ], bbox_params=A.BboxParams(format='yolo'))

  # Create a dataset
  data = YoloDataset(
    path_to_data,
    subdir=subdir,
    transform=tnsfm,
    resize_size=resize,
    S=grid
  )
  
  data_train, data_val = random_split(
    data,
    [1 - test_size, test_size],
    generator=torch.Generator().manual_seed(random_seed)
  )
  
    # Create a DataLoader
  if "cuda" in pin_memory_device.type:
    data_train = DataLoader(dataset=data_train, batch_size=batch_size, num_workers=workers, shuffle=True, pin_memory=True, pin_memory_device=pin_memory_device.type)
    data_val = DataLoader(dataset=data_val, batch_size=batch_size, num_workers=workers, shuffle=True, pin_memory=True, pin_memory_device=pin_memory_device.type)

  else:
    data_train = DataLoader(dataset=data_train, batch_size=batch_size, num_workers=workers)
    data_val = DataLoader(dataset=data_val, batch_size=batch_size, num_workers=workers)

  return data_train, data_val
  

def YoloLoaderTest(
    path_to_data: str, 
    subdir: str,
    batch_size: int, 
    resize: None|int|tuple|list = None, 
    grid: int = 7, 
    workers: int = 4, 
    pin_memory_device: None|object = None
  ):
  """
    Load data from the Dataset and batch it into a DataLoader.

    Args:
    path_to_data: str: Path to the directory containing subdirectories with images.
    batch_size: int: Batch size.
    resize: Union[None, int, tuple, list]: Resize images to this size.
    grid: int: Grid size. Same for each dimension.
    workers: int: Number of workers.
    pin_memory_device: Union[None, object]: Device to pin memory to.
    
    Returns:
    test_data: DataLoader: Test data.
  """
  
  # Create a dataset
  data = YoloDataset(
    path_to_data,
    subdir=subdir,
    resize_size=resize,
    S=grid
  )
    
    # Create a DataLoader
  if "cuda" in pin_memory_device.type:
    data = DataLoader(dataset=data, batch_size=batch_size, num_workers=workers, pin_memory=True, pin_memory_device=pin_memory_device.type)
  else:
    data = DataLoader(dataset=data, batch_size=batch_size, num_workers=workers)

  return data