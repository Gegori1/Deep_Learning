# %% libraries
import os
import sys
import shutil
import numpy as np

from google.colab import drive
drive.mount('/content/drive')

# %% parameters

src_name: str = 'color_2'
dst_name: str = 'test'
path_to_data: str = '/content/drive/My Drive/Deep_Learning_class/plant_illness/'

# %% move files back

src_path = os.path.join(path_to_data, src_name)
dst_path = os.path.join(path_to_data, dst_name)

for folder in os.listdir(dst_path):
    src_folder = os.path.join(src_path, folder)
    dst_folder = os.path.join(dst_path, folder)
    for file in os.listdir(dst_folder):
        src_file = os.path.join(dst_folder, file)
        dst_file = os.path.join(src_folder, file)
        shutil.move(dst_file, src_file)