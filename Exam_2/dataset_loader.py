import glob
from cv2 import (
  VideoCapture,
  resize,
  CAP_PROP_FRAME_COUNT,
  CAP_PROP_POS_FRAMES
)
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import numpy as np


class VideoDataset(Dataset):
  def __init__(self, root: str, frames:int, resize_size: int, transform: None|object=None):
    self.root = root
    self.k = frames
    self.resize = resize_size
    self.transform = transform

    self.paths = glob.glob(root + '*/*')
    self.target_encode = [i.split('/')[-2] for i in self.paths]
    self.classes = sorted(list(set(self.target_encode)))
    self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
    self.targets = [self.class_to_idx[c] for c in self.target_encode]
    self.samples = list(zip(self.paths, self.targets))
    self.n_samples = len(self.paths)

  def crop_csquare(self, frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]

  def load_images(self, video_path):
      k = self.k
      video = VideoCapture(video_path)
      video_frames = int(video.get(CAP_PROP_FRAME_COUNT))
      if video_frames < k:
          raise ValueError(
              f"The number of frames per video ({video_frames}) must be greater than or equal to the taken sample ({k})"
          )
      sample = np.sort(np.random.choice(range(video_frames), k, replace=False))

      frames = []
      for i in sample:
          video.set(CAP_PROP_POS_FRAMES, i)
          ret, frame = video.read()
          if not ret:
              break
          frame = self.crop_csquare(frame)
          frame = resize(frame, (self.resize, self.resize))
          frame = frame.transpose(2, 0, 1)
          frames.append(frame)

      video.release()
      return np.array(frames)

  def __len__(self):
    return self.n_samples

  def __getitem__(self, idx):
    path, target = self.samples[idx]
    img_sample = self.load_images(path)

    if isinstance(img_sample, torch.Tensor):
        img_sample = img_sample.float()
    elif isinstance(img_sample, np.ndarray):
      img_sample = torch.from_numpy(img_sample)
      img_sample = img_sample.float()
    else:
      raise ValueError("Invalid data type")

    if self.transform:
      img_sample = self.transform(img_sample)
      
      sample = {"sequence": img_sample, "target": target}

    return sample