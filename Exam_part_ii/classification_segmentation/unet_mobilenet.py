# %%
import torch
from torch import nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchsummary import summary
# %%

name_pretrained_unet = 'mateuszbuda/brain-segmentation-pytorch'

mobilenet1 = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
mobilenet2 = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)



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
    def __init__(self, in_channels, num_classes: int, seg_channels: int=1):
        super().__init__()
        self.in_channels = in_channels
        self.classes = num_classes
        self.seg_channels = seg_channels
        
        # unet. Radiography input (1 channel) but  pretrained has 3 channels
        self.unet = torch.hub.load(
            repo_or_dir=name_pretrained_unet, model='unet',
            in_channels=3, out_channels=self.seg_channels,
            init_features=32, pretrained=True
        )
        self.unet.encoder1[0] = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=32, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False
        )
        # mobilenet
        self.mobile_net = list(mobilenet1.children())[:-2][0]
        # output of mobilenet
        self.mobile_net[0][0] = nn.Conv2d(
            in_channels=self.in_channels, 
            out_channels=16, 
            kernel_size=3, 
            stride=2, 
            padding=1, 
            bias=False
        )
        
        # mobilenet after unet
        self.mobile_unet = list(mobilenet2.children())[:-2][0]
        # output of mobilenet after unet
        self.mobile_unet[0][0] = nn.Conv2d(
            in_channels=self.seg_channels, 
            out_channels=16, 
            kernel_size=3, 
            stride=2, 
            padding=1, 
            bias=False
        )
        # classifier
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=576*2, out_features=1024),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1024, out_features=3)
        )
        
        self.flatten = nn.Flatten()
        
        
        
        
    def forward(self, x):
        
        x1 = self.unet(x)
        x1_ = self.mobile_unet(x1)

        x2 = self.mobile_net(x)

        x3 = torch.cat((x1_, x2), dim=1)
        x3 = self.avgpool(x3)
        x3 = self.flatten(x3)
        x3 = self.classifier(x3)
        
        return x1, x3
        
        
# %% 
if __name__ == '__main__':
    model = NetCovid(in_channels=1, num_classes=3, seg_channels=1)

    X = torch.randn(3, 1, 256, 256)
    
    out1, out2 = model(X)

    summary(model, (1, 3256, 256))

# %%
