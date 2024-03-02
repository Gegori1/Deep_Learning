# %% libraries
import os
import sys
import shutil
import numpy as np

from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2 as transforms
from torch.utils.data import Subset
from custom_load import LeafDataset


from google.colab import drive
drive.mount('/content/drive')

# %% parameters

path_to_data: str = '/content/drive/My Drive/Deep_Learning_class/plant_illness/color_2/'

# %% train and eval data loader

def load_train_eval_data(path_to_data: str, batch_size: int = 32, eval_size: float = 0.2, resize: int = 225, random_state: int = 42):
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
    # Create a transformation
    trnsf = transforms.Compose([
        transforms.RandomCrop(size=resize),
        transforms.Resize([resize, resize]),
        transforms.RandomAdjustSharpness(2),
        transforms.RandomAutocontrast(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)
    ])

    # Create a dataset
    data = ImageFolder(root=path_to_data, transform=trnsf)
    
    targets = data.targets

    # Perform a stratified split
    train_data, eval_data = train_test_split(data, test_size=eval_size, stratify=targets, random_state=random_state)
    stratified_train_data = Subset(data, train_data)
    stratified_eval_data = Subset(data, eval_data)
    
    # set a weighted sampler for the DataLoader
    count_class = dict(Counter(data.targets))
    weights = [1.0 / v for v in count_class.values()]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    
    train_data = DataLoader(dataset=stratified_train_data, batch_size=batch_size, sampler=sampler, num_workers=1)
    eval_data = DataLoader(dataset=stratified_eval_data, batch_size=batch_size, sampler=sampler, num_workers=1)
    
    return train_data, eval_data


# %% test data loader

def load_test_data(path_to_data: str, batch_size: int = 32, resize: int = 225):
    """
    Load test data from a directory containing subdirectories with images.

    Args:
    path_to_data: str: Path to the directory containing subdirectories with images.
    
    Returns:
    test_data: DataLoader: Test data.
    """

    # Create a dataset
    data = ImageFolder(root=path_to_data)
    
    # Create a DataLoader
    test_data = DataLoader(dataset=data, batch_size=batch_size, shuffle=True, num_workers=1)
    
    return test_data