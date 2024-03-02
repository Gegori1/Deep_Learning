# libraries
import os
import numpy as np
from google.colab import drive
drive.mount('/content/drive')

# parameters
path_to_data: str = '/content/drive/My Drive/Deep_Learning_class/plant_illness/'
top: int = 2000
folder_to_check: str = 'color_2'


path_folders = os.path.join(path_to_data, folder_to_check)
folders = os.listdir(path_folders)

os.mkdir(os.path.join(path_to_data, folder_to_check + '_top'))
    

# remove till top
for folder in folders:
    files = os.listdir(os.path.join(path_folders, folder))
    len_folder = len(files)
    if len_folder > top:
        # create folder in top
        folder_top = os.path.join(path_to_data, folder_to_check + '_top', folder)
        os.mkdir(folder_top)
        # move top files
        files_rm = np.random.choice(files, int(len_folder - top), replace=False)
        for file in files_rm:
            file_path = os.path.join(path_folders, folder, file)
            new_file_path = os.path.join(folder_top, file)
            os.rename(file_path, new_file_path)
            print(f'{file} moved to {folder_top}')
        
        
