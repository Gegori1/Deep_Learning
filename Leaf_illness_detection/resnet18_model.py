import torch
from torch import nn
from typing import Type

class BlockModel18(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        stride: int = 1, # this value will be chaned to 2
        expansion: int = 1, # standard in resnet18 and resnet34. resnet50 uses 4
        identity_downsample: nn.Module = None,
        ) -> None:
        "Define the block sublayers"
        super().__init__()
        self.dropout_rate = 0.5
        self.identity_downsample = identity_downsample
        # kernel = 3, padding = 1 to same. stride = 2 to downsample
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels*expansion, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_rate)
        

    def forward(self, inp: torch.Tensor) -> torch.Tensor:

        out = self.conv1(inp)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.identity_downsample is not None:
            # identity block for layer 1
            # convolutional layer layer 2 to 4 to match
            inp = self.identity_downsample(inp)

        out += inp
        out = self.relu(out)
        return out

class ResNet34(nn.Module):
    def __init__(
        self,
        img_channels: int,
        num_classes:int,
        layers: int,
        # block: Type[BlockModel18, BlockModel50]
        block: Type[BlockModel18],
    ) -> None:
        super().__init__()
        if layers == 18:
            layers = [2, 2, 2, 2]
            self.expansion = 1
            
        elif layers == 34:
            layers = [3, 4, 6, 3]
            self.expansion = 1
            
        elif layers == 50:
            layers = [3, 4, 6, 3]
            self.expansion = 4
            
        else:
            raise ValueError("Layers must be 18 or 34")
            
        self.in_channels = 64          
        
        # First layer
        self.PrincipalBlock = nn.Sequential(
            nn.Conv2d(img_channels, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # dense layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512*self.expansion, num_classes)
        
    def _make_layer(self, block, out_channels, num_residual_blocks, stride):
        
        donwsample = None
        if stride != 1 or (self.in_channels != out_channels * self.expansion):
            donwsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * self.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels*self.expansion)
            )
        
        layers = []
        # first block is added to the layer
        layers.append(block(self.in_channels, out_channels, stride, self.expansion, donwsample))
        # taking advantage of the stack, the in_channels will be updated
        self.in_channels = out_channels * self.expansion
        
        # add the rest of the blocks
        # These blocks do not need to downsample
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, out_channels, expansion=self.expansion))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.PrincipalBlock(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        
        return x