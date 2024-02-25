# Make sure to adjust the `dataset_dir` and `output_dir` to your actual dataset and desired output directory. The `transform` should also be defined according to your augmentation needs.

# This script assumes that the dataset is structured with one directory per class, each containing the respective images. The `ImageFolder` class provided by `torchvision.datasets` is a convenient way to load such datasets.

# Note that this script will create a lot of augmented images to balance the classes. Be careful with the number of generated images, as this might lead to overfitting if the augmentations are not diverse enough or if the original dataset is small. It might be better to augment images on-the-fly during training instead of pre-generating a laYrge number of augmented images.

# You're correct that augmenting images on-the-fly without addressing class imbalance might still result in certain classes being underrepresented during training. However, the purpose of on-the-fly augmentation is not to balance the dataset but to introduce variability to the training examples, which helps prevent overfitting and allows the model to generalize better.

# To address class imbalance when augmenting images on-the-fly, you can use techniques such as weighted random sampling or oversampling the minority classes during training. Here's how you could implement weighted random sampling in PyTorch:

# 1. Calculate weights for each class based on their frequency in the dataset.
# 2. Use these weights to create a sampler that will be used by the DataLoader to ensure each batch of data has a balanced class distribution.

# Here's an example of how to implement this:

# ```python
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

# Define your dataset directory
dataset_dir = 'path/to/your/dataset'

# Define the transformation/augmentation you want to apply
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    # ... other transformations as needed
])

# Load your dataset
dataset = datasets.ImageFolder(root=dataset_dir, transform=transform)

# Count the number of images per class
class_counts = {}
for _, index in dataset:
    label = dataset.classes[index]
    class_counts[label] = class_counts.get(label, 0) + 1

# Calculate weights for each class
class_weights = {class_label: max(class_counts.values()) / count
                 for class_label, count in class_counts.items()}

# Create a list of weights for each sample in the dataset
sample_weights = [class_weights[dataset.classes[label]] for _, label in dataset.samples]

# Create a weighted random sampler
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

# Create a DataLoader with the weighted sampler
data_loader = DataLoader(dataset, batch_size=32, sampler=sampler)

# Now, when you iterate over the DataLoader, each batch should have a balanced class distribution
for images, targets in data_loader:
    # Train your model here
    pass
# ```

# By using `WeightedRandomSampler`, the DataLoader will sample images from the dataset with a probability proportional to the specified sample weights, which helps to balance the classes in each batch. Note that the `replacement` parameter is set to `True` to allow for the possibility of sampling the same image more than once in an epoch, which is typically necessary when dealing with severe class imbalances.