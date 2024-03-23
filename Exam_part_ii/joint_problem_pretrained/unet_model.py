# %%

import torch
from torch import nn
from torchsummary import summary

# %%

class CustomUnet(nn.Module):
    """
    loads pretrained unet and modifies the number of input channels
    """
    
    def __init__(self, in_channels: int=1) -> None:
        super().__init__()
        self.in_channels = in_channels
        
        self.unet = torch.hub.load(
            repo_or_dir='mateuszbuda/brain-segmentation-pytorch', model='unet', 
            in_channels=3, out_channels=1, init_features=32, pretrained=True
        )
        
        self.unet.encoder1[0] = nn.Conv2d(
            in_channels=self.in_channels, 
            out_channels=32, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unet(x)
    
    
# %%
if __name__ == '__main__':
    X = torch.randn(3, 1, 256, 256)
    model = CustomUnet(in_channels=1)
    out = model(X)
    print(out.shape)
    summary(model, (1, 256, 256))


# %%
