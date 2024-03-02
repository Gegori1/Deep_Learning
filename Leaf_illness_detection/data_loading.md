To achieve the tasks you've mentioned using PyTorch, you'll need to follow these steps:

1. Load your images using `ImageFolder`.
2. Split your dataset into test and evaluation sets with a random seed.
3. Create weighted sampling for your DataLoader to ensure that each batch has a balanced number of samples from each class.

Here's how you can do this:

1. **Load images using `ImageFolder`:**
```python
import torch
from torchvision import datasets, transforms

# Define your data transform to apply to each image
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to a fixed size (modify as needed)
    transforms.ToTensor()  # Convert images to PyTorch tensors
])

# Load images using ImageFolder
dataset = datasets.ImageFolder(root='path_to_images', transform=transform)
```

2. **Split the dataset:**
```python
from torch.utils.data import random_split

# Set the random seed for reproducibility
torch.manual_seed(42)

# Define the sizes of your splits, e.g., 80% for training and 20% for evaluation
train_size = int(0.8 * len(dataset))
eval_size = len(dataset) - train_size

# Split the dataset
train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])
```

3. **Create weighted sampling for DataLoader:**
```python
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

# Count the number of occurrences of each class in the dataset
class_counts = {}
for _, index in train_dataset:
    label = dataset.classes[index]
    if label in class_counts:
        class_counts[label] += 1
    else:
        class_counts[label] = 1

# Calculate weights for each class
weights = [1.0 / class_counts[dataset.classes[idx]] for _, idx in train_dataset]

# Create a WeightedRandomSampler
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

# Create your DataLoaders with the samplers
train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=True)  # Assuming you want to shuffle the eval set
```

Please note that in the case of `WeightedRandomSampler`, weights should be a sequence where each element is the weight of the corresponding sample. In the example above, `weights` is calculated based on the inverse frequency of the class occurrence. Also, `replacement` is set to `True` to allow for sampling the same data point multiple times.

For the evaluation set, you might want to weigh the samples similarly, or you might simply want to shuffle them as usual. The above example shows a simple shuffle without weighted sampling for the evaluation set. If you need weighted sampling for the evaluation set as well, you would follow a similar process as for the training set.

Remember to replace `'path_to_images'` with the actual path to your images directory. Adjust the batch sizes (`batch_size=32`) and any transformation parameters as per your requirements.