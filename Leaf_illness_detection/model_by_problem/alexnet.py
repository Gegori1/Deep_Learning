# %%
import torch
from torch import nn

# %%
class AlexNet(nn.Module):
    def __init__(self, input_channels: int, num_classes:int):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        
        self.relu = nn.ReLU()
        
        self.conv1 = nn.Conv2d(in_channels=self.input_channels, out_channels=64, kernel_size=11, stride=4, padding=2)
        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, padding=2)
        self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.mpool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.flatten = nn.Flatten()
        
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(256*6*6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, self.num_classes)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.mpool1(x)
        x = self.relu(self.conv2(x))
        x = self.mpool2(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.mpool3(x)
        
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

# %%
# %%
if __name__ == "main":
    X = torch.randn(1, 3, 224, 224)
    model = AlexNet(input_channels=3, num_classes=38)
    out = model(X)

    print(out.shape)
# %%
