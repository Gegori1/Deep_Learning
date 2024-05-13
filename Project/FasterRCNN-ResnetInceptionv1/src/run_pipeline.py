# Install libraries
! pip install facenet_pytorch

# Import libraries
from google.colab import drive
drive.mount('/content/drive')

import os
import sys
import numpy as np

root_dir = '...'

lib_dir = os.path.join(root_dir, 'src')
sys.path.append(lib_dir)
# from ... import CustomDataset
# from ... import get_transforms
# from models import FasterRCNN, FaceNet
# from pipeline import Pipeline

# Import data
path_to_drive_data = os.path.join(root_dir, 'data/')
path_to_zip = os.path.join(path_to_drive_data, 'Archive_classification.zip')
current_folder_path = '/content/localdata/'

!unzip -q $path_to_zip -d $current_folder_path

# Load data
dataset = CustomDataset(path_to_data=current_folder_path, transform=get_transforms())

# define parameters
num_classes = 2
model_fasterrcnn_load_name = '...'

# Load embeddings
# - If embeddings are already saved, load them
# - If embeddings are not saved, run the pipeline
if os.path.exists(os.path.join(current_folder_path, 'embeddings.npy')):
    embeddings = np.load(os.path.join(current_folder_path, 'embeddings.npy'))
else:
    embeddings = None


# Instantiate models
# - Model 1
state_faster_rcnn = torch.load(os.path.join(root_dir, 'models', model_fasterrcnn_load_name))
model_fasterrcnn = FasterRCNN(num_classes=num_classes)
# - Model 2
model_facenet = FaceNet()

# Run pipeline
pipe_line_face = Pipeline(model_fasterrcnn, model_facenet, dataset, k)

pipe_line_face.print_results()

# Save embedding if needed
save_embeddings = False
if save_embeddings:
    pipe_line_face.save_embeddings()

