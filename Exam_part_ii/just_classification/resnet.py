# %%
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights
from torchsummary import summary
# %%

resnet = resnet18(weights=ResNet18_Weights.DEFAULT)

# %%

class ResNetCovid(nn.Module):
    def __init__(self, in_channels, num_classes: int):
        super().__init__()
        self.in_channels = in_channels
        self.classes = num_classes
        self.resnet = resnet
        self.resnet.conv1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=64, 
            kernel_size=7, 
            stride=2, 
            padding=3, 
            bias=False
        )
        self.resnet.fc = nn.Linear(512, self.classes)
        
    def forward(self, x):
        return self.resnet(x)
    
# %%