import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from custom_load import LeafDataset

dataset_dir = 'data/leaf_illness_dataset'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

classes = {'healthy': 0, 'ill': 1}

train_dataset = LeafDataset(root=dataset_dir, classes=classes, transform=transform)
valid_dataset = LeafDataset(root=dataset_dir, classes=classes, transform=transform)
test_dataset = LeafDataset(root=dataset_dir, classes=classes)

