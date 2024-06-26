# %%
from torch import nn
from torchsummary import summary
# %% import pretrained
from torchvision.models import resnet18, ResNet18_Weights
# # from torchvision.models.detection.retinanet import RetinaNet
# from torchvision.models.detection import retinanet_resnet50_fpn
# %% custom yolo model
class YoloModel(nn.Module):
  """
  This YOLO model is a custom implementation to predict a single class and bounding box.
  Therefore no iou threshold or confidence threshold is used.
  
  Args:
    grid_size (int): The size of the grid to divide the image into.
  
  Returns:
    bounding_box (list): The bounding box coordinates of the object.
  """
  def __init__(self, grid_size=7, bounding_box=2):
    super().__init__()
    
    self.grid_size = grid_size
    self.bounding_box = bounding_box
    # Define model
    resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
    
    resnet = list(resnet.children())[:-2]
    # average pool based on grid size
    average_pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))
    # flatten output. Later we will reshape it to (grid_size, grid_size, 5)
    resnet.append(average_pool)
    fc = nn.Sequential(
      nn.Flatten(start_dim=1),
      nn.Linear(512 * grid_size * grid_size, 992),
      nn.Dropout(0.5),
      nn.ReLU(0.1),
      nn.Linear(992, bounding_box * grid_size * grid_size * 5),
    )
    resnet.append(fc)
    self.resnet = nn.Sequential(*resnet)
    
  def forward(self, x):
    x = self.resnet(x)
    return x
    

# %%
if __name__ == '__main__':
  import torch
  X = torch.randn((3, 3, 224, 224))
  model = YoloModel(grid_size=4, bounding_box=2)
  # out = model(X)
  # print(out.shape)
  summary(model, (3, 224, 224))
# %%
