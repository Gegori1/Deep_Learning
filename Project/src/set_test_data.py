# %% libraries
import os
# import sys
import shutil
import numpy as np


# %% parameters
class TestData:
    def __init__(self, img_to_test: int, src_name: str, dst_name: str, path_to_data: str):
        self.img_to_test = img_to_test
        self.src_name = src_name
        self.dst_name = dst_name
        self.path_to_data = path_to_data

    def remove_files(self):
        path_src = os.path.join(self.path_to_data, self.src_name)
        files = [file for file in os.listdir(path_src) if os.path.isfile(os.path.join(path_src, file))]
        for file in files:
            file_path = os.path.join(path_src, file)
            os.remove(file_path)

    def create_folders(self):
        folders = os.listdir(os.path.join(self.path_to_data, self.src_name))
        test_folder = os.path.join(self.path_to_data, self.dst_name)
        if not os.path.exists(test_folder):
            os.makedirs(test_folder)
        for folder in folders:
            folder_path = os.path.join(test_folder, folder)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

    def populate_test(self, seed=42):
        folders = os.listdir(os.path.join(self.path_to_data, self.src_name))
        test_folder = os.path.join(self.path_to_data, self.dst_name)
        for folder in folders:
            folder_dst = os.path.join(test_folder, folder)
            len_files_dst = len(os.listdir(folder_dst))
            if len_files_dst < self.img_to_test:
                folder_src = os.path.join(self.path_to_data, self.src_name, folder)
                files_src = os.listdir(folder_src)
                rng = np.random.default_rng(seed=seed)
                files_src = rng.choice(files_src, self.img_to_test - len_files_dst, replace=False)
                for file in files_src:
                    src = os.path.join(folder_src, file)
                    dst = os.path.join(folder_dst, file)
                    shutil.move(src, dst)
                    
    def rename_source_folder(self, new_folder_name: str):
        path_src = os.path.join(self.path_to_data, self.src_name)
        new_path_src = os.path.join(self.path_to_data, new_folder_name)
        os.rename(path_src, new_path_src)
        self.src_name = new_folder_name






