# %%
import torch
from torch import nn
from torchsummary import summary

from torchvision.transforms import transforms

from resnet18_dual_model import ResNet18_dual
from unet_model import CustomUnet

# %%
class NetCovid(nn.Module):
    """
    NN to get semantic segmentation and covid, normal or non-covid classification
    
    Args:
    - um_classes: int, number of classes to classify
    - seg_channels: int, number of channels in the segmentation output
    
    Returns:
    - Tensor, segmentation output
    - Tensor, classification output

    """
    def __init__(self, in_channels: int, num_classes: int, seg_channels: int=1, unet_path: str=..., resnet_path: str=...):
        super().__init__()
        self.in_channels = in_channels
        self.classes = num_classes
        self.seg_channels = seg_channels
        
        # pretrained unet
        self.unet_model = CustomUnet(in_channels=self.in_channels)
        self.unet = self.unet_model.load_state_dict(torch.load(unet_path)["model"])
        
        # dual pretrained resnet18
        self.resnets = ResNet18_dual(img_channels=self.in_channels, num_classes=self.classes)
        self.resnets.load_state_dict(torch.load(resnet_path)["model"])
        
        self.resize = transforms.Resize((224, 224))
        
        
    def forward(self, x):
        
        x_seg = self.unet_model(x)

        x_seg_re = self.resize(x_seg)

        x = self.resize(x)

        x_class = self.resnets(x, x_seg_re)
        
        return x_seg, x_class
        
        
# %% 
if __name__ == '__main__':
    model = NetCovid(in_channels=1, num_classes=3, seg_channels=1)

    X = torch.randn(3, 1, 256, 256)
    
    out1, out2 = model(X)
    
    print(out1.shape, out2.shape)

# %%
