from facenet_pytorch import InceptionResnetV1
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# ==============
# Faster R-CNN
# ==============

def get_fasterrcnn(num_classes):
  """
    Loads Pretrained Faster R-CNN model from torchvision.
    No inputs are transformed.
  """
  model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
  in_features = model.roi_heads.box_predictor.cls_score.in_features
  model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

  return model

# ==============
# Facenet
# ==============

def get_facenet():
  """
    Loads Pretrained FaceNet model from facenet_pytorch.
  """
  model = InceptionResnetV1(pretrained='vggface2').eval()

  return model