import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

def get_processed_dataset_path(raw_dataset_path):
    """
    Returns a path to the processed data folder, e.g.

    raw_dataset_path = /mnt/disks/disk1/data/raw/rgb/
    processed_dataset_path = /mnt/disks/disk1/data/processed/rgb/
    """
    return raw_dataset_path.replace("raw", "processed")


def get_splits(vids):
  pass

def preprocessRGB(raw_dataset_path, resize_dims):
  """ 
  Preprocess RGB files in the raw dataset path and 
  save an index of the contents to CSV
  
  @param raw_dataset_path File path to RGB JPG files
  @param resize_dims Dimensions to resize raw images [height, width]
  """ 
  processed_dataset_path = get_processed_dataset_path(raw_dataset_path)
  if resize_dims is None:
    resize_dims = [720, 1280]

  # list files in folder
  files = list()
  for (dirpath, dirnames, filenames) in os.walk(raw_dataset_path):
    files += [os.path.join(dirpath, file) for file in filenames]

  # create pandas data frame with RGB data
  rgb = pd.DataFrame(rgb_files, columns=["filename"])
  regex = rgb['filename'].str.extract(
    '\/(?P<dance>\w+)\/(?P<vid>[^/]+)_(?P<start_fid>[0-9]+)_(?P<relative_fid>[0-9]+)\.jpg', 
    flags=0, 
    expand=True)
  rgb = rgb.join(regex).dropna(axis=0, subset=["dance"])

  # label dances with unique identifiers (only for the 10 original in the paper)
  dance_dict = {
      'ballet': 0, 'break': 1, 'flamenco': 2, 'foxtrot': 3, 'latin': 4,
      'quickstep': 5, 'square': 6, 'swing': 7, 'tango': 8, 'waltz': 9
  }
  rgb['dance_id'] = rgb['dance'].apply(lambda x: dance_dict.get(x))
  rgb.dropna(axis=0, subset=["dance_id"], inplace=True)
  rgb.reset_index(drop=True, inplace=True)

  # add filepath for processed data
  rgb['processed_path'] = os.path.join(
    processed_dataset_path, 
    os.path.basename(rgb['filepath'])[:-4], ".npy")

  # split to train, val, test, and save file indexes
  rgb.to_csv("data/rgb_index.csv")

  # create a transform 
  transform = transforms.Compose([
      transforms.Resize(resize_dims),
      transforms.ToTensor(),
      transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225])])

  # transform raw data and save processed files
  for i, fpath in enumerate(files):
    # load and transform image
    X = Image.open(fpath)
    X = transform(X)
    # get class
    y = rgb["dance_id"][i]
    # save to file
    obs = np.asarray([X, y])
    np.save(rgb['processed_path'][i], obs, allow_pickle=True)


class cnnDataset(Dataset):
    """ Custom dataset for CNN image data
    """
    def __init__(self, index_filepath):
      self.file_index =  pd.read_csv(index_filepath, index_col=0)
        
    def __len__(self):
      """ Return number of obs in the dataset
      """
      return len(self.file_index)

    def __getitem__(self, index):
      """ Return X, y for a single observation
      """
      # get filepath 
      path = self.file_index["processed_path"].iloc[index]
      X, y = np.load(path, allow_pickle=True)
      return X, y
