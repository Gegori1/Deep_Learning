# %% libraries
import os
import sys
import shutil

import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms

from google.colab import drive
drive.mount('/content/drive')

# %% parameters

img_to_test: int = 30
src_name: str = 'color_2'
dst_name: str = 'test'
path_to_data: str = '/content/drive/My Drive/Deep_Learning_class/plant_illness/'

# %% create test folders

path_src = os.path.join(path_to_data, src_name)
folders = os.listdir(path_src)

test_folder = os.path.join(path_to_data, dst_name)

if not os.path.exists(test_folder):
    os.makedirs(test_folder)

for folder in folders:
    folder_path = os.path.join(test_folder, folder)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        


# %% Move images to test folders

for folder in folders:
    # check if the test folder is less than 30
    folder_dst = os.path.join(test_folder, folder)
    len_files_dst = len(os.listdir(folder_dst))
    if len_files_dst < 30:
        folder_src = os.path.join(path_src, folder)
        files_src = os.listdir(folder_src)
        files_src = np.random.choice(files_src, img_to_test - len_files_dst, replace=False)
        
        for file in files_src:
            src = os.path.join(folder_src, file)
            dst = os.path.join(folder_dst, file)
            shutil.move(src, dst)






