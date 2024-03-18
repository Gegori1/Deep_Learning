import os
import shutil
import numpy as np

import torch
from torchvision import datasets, transforms

from google.colab import drive
drive.mount('/content/drive')

path_to_data = '/content/drive/My Drive/Deep_Learning_class/plant_illness/'

path_src = os.path.join(path_to_data, 'color_2')
path_dst = os.path.join(path_to_data, 'eval')

folders = os.listdir(path_src)

# Create eval folders
if not os.path.exists(path_dst):
    os.makedirs(path_dst)
    
for folder in folders:
    folder_path = os.path.join(path_dst, folder)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

# Move images to eval folders
for folder in folders:
    folder_dst = os.path.join(path_dst, folder)
    folder_src = os.path.join(path_src, folder)
    
    folder_src_files = os.listdir(folder_src)
    folder_dst_files = os.listdir(folder_dst)
    
    folder_dst_len = len(folder_src_files)
    folder_src_len = len(folder_dst_files)
    
    if folder_src_len == 0:
        print(f'Source foulder: {folder} is empty')
        continue
    
    
    files_to_move = np.random.choice(folder_src_files, int(folder_src_len * 0.2), replace=False)
    
    for file in files_to_move:
        src = os.path.join(folder_src, file)
        dst = os.path.join(folder_dst, file)
        shutil.move(src, dst)
    
    # files_ratio: int = int(folder_dst_len / folder_src_len * 100)
    # if files_ratio < 20:
    #     files_to_move = np.random.choice(folder_src_files, 20 - folder_dst_len, replace=False)
        
    #     for file in files_to_move:
    #         src = os.path.join(folder_src, file)
    #         dst = os.path.join(folder_dst, file)
    #         shutil.move(src, dst)   
    
    # folder_src = os.path.join(path_src, folder)
    # files_src = os.listdir(folder_src)
    
    # files_src = np.random.choice(files_src, 30, replace=False)
    
    # for file in files_src:
    #     src = os.path.join(folder_src, file)
    #     dst = os.path.join(path_dst, folder, file)
    #     shutil.move(src, dst)
    
