import glob
from cv2 import imread, resize, cvtColor, COLOR_BGR2RGB
import torch
from torch.utils.data import Dataset

class YoloDatasetTest(Dataset):
    def __init__(self, root: str, resize_size: int)-> None:
        super().__init__()
        self.root_dir = root
        self.resize_size = resize_size
        self.data = glob.glob(self.root_dir + "*jpg")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data[idx]
        img = imread(img_path)
        img = cvtColor(img, COLOR_BGR2RGB)
        img = resize(img, self.resize_size)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        img = img / 255
        
        sample = {"image": img}
        
        return sample