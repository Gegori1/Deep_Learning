# %% libraries
import torch
from torch.utils.data import DataLoader
from dataset_loader import VideoDataset
from torchvision.transforms import transforms
from torch.utils.data import random_split


# %% train and eval data loader

def load_train_eval_data(
    path_to_data: str, 
    batch_size: int=32, 
    eval_size: float=0.2, 
    resize: int=224, 
    nframes: int= 10, 
    random_state: int=42, 
    workers: int=4, 
    pin_memory_device: None|object=None
  ):
    """
    Load train and evaluation data from a directory containing subdirectories with images.

    Args:
    path_to_data: str: Path to the directory containing subdirectories with images.
    eval_size: float: Size of the eval set.
    random_state: int: Random state for the train_eval_split function.
    
    Returns:
    train_data: DataLoader: Train data.
    eval_data: DataLoader: eval data.
    """
    
    trnsf = transforms.Compose([
        # transforms.RandomAdjustSharpness(2),
        # transforms.RandomAutocontrast(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.Normalize(mean=0.5, std=0.5)
    ])

    # Create a dataset
    dataset = VideoDataset(
        root=path_to_data,
        frames=nframes,
        resize_size=resize,
        transform=trnsf
      )

    subset_train, subset_eval = random_split(dataset, [1 - eval_size, eval_size], generator=torch.Generator().manual_seed(random_state))

    if "cuda" in pin_memory_device.type:
      train_data = DataLoader(dataset=subset_train, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, pin_memory_device=pin_memory_device.type)
      eval_data = DataLoader(dataset=subset_eval, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, pin_memory_device=pin_memory_device.type)
    else:
        train_data = DataLoader(dataset=subset_train, batch_size=batch_size, shuffle=True, num_workers=workers)
        eval_data = DataLoader(dataset=subset_eval, batch_size=batch_size, shuffle=True, num_workers=workers)

    return train_data, eval_data


# %% test data loader


def load_test_data(path_to_data: str, batch_size: int = 32, resize: int = 224, nframes: int=10, workers: int = 4, pin_memory_device: None|object = None):
  """
    Load test data from a directory containing subdirectories with images.

    Args:
    path_to_data: str: Path to the directory containing subdirectories with images.
    
    Returns:
    test_data: DataLoader: Test data.
  """

  trnsf = transforms.Compose([
          transforms.Normalize(mean=0.5, std=0.5)
    ])

  # Create a dataset
  dataset = VideoDataset(
      root=path_to_data,
      frames=nframes,
      resize_size=resize,
      transform=trnsf
    )
    
    # Create a DataLoader
  if "cuda" in pin_memory_device.type:
    test_data = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, pin_memory_device=pin_memory_device.type)
  else:
    test_data = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    
  return test_data