import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import json

def get_processed_dataset_path(raw_dataset_path):
    """
    Returns a path to the processed data folder, e.g.

    raw_dataset_path = /mnt/disks/disk1/data/raw/rgb/
    processed_dataset_path = /mnt/disks/disk1/data/processed/rgb/
    """
    return raw_dataset_path.replace("raw", "processed")


def get_splits(vids):
  """
  Split list of video IDs into train (80%), validation (10%), and test (10%).
  Sorts the list of video IDs and sets a seed so that it is reproducible.

  @param vids List of video IDs
  @return (train_vids, val_vids, test_vids) 80-10-10 split of video IDs
  """
  vids = sorted(vids)
  train_vids, test_vids = train_test_split(vids, test_size=0.2, random_state=1)
  test_vids, val_vids = train_test_split(test_vids, test_size=0.5, random_state=1)
  return (train_vids, val_vids, test_vids)


def preprocessRGB(raw_dataset_path, resize_dims=[720,1280]):
  """ 
  Preprocess RGB files in the raw dataset path and 
  save an index of the contents to CSV
  
  @param raw_dataset_path File path to RGB JPG files
  @param resize_dims Dimensions to resize raw images [height, width]
  """ 
  processed_dataset_path = get_processed_dataset_path(raw_dataset_path)
    
  # list files in folder
  rgb_files = list()
  for (dirpath, dirnames, filenames) in os.walk(raw_dataset_path):
    rgb_files += [os.path.join(dirpath, file) for file in filenames]

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
  rgb['processed_path'] = rgb['filename'].apply(lambda x: os.path.join(
    processed_dataset_path, f"{os.path.basename(x)[:-4]}.npy"))
  
  # split to train, val, test, and save file indexes
  train_vids, val_vids, test_vids = get_splits(rgb['vid'].drop_duplicates())
  train = rgb[rgb['vid'].isin(train_vids)].reset_index(drop=True)
  val = rgb[rgb['vid'].isin(val_vids)].reset_index(drop=True)
  test = rgb[rgb['vid'].isin(test_vids)].reset_index(drop=True)
  train.to_csv(os.path.join(os.path.dirname(processed_dataset_path),"rgb_train_index.csv"))
  val.to_csv(os.path.join(os.path.dirname(processed_dataset_path),"rgb_val_index.csv"))
  test.to_csv(os.path.join(os.path.dirname(processed_dataset_path),"rgb_test_index.csv"))
    
  # create a transform 
  transform = transforms.Compose([
      transforms.Resize(resize_dims),
      transforms.ToTensor(),
      transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225])])

  # transform raw data and save processed files
  for i, fpath in enumerate(rgb_files):
    # load and transform image
    X = Image.open(fpath)
    X = transform(X)
    # get class
    y = rgb["dance_id"][i]
    # save to file
    obs = np.asarray([X.numpy(), y])
    np.save(rgb['processed_path'][i], obs, allow_pickle=True)


def preprocessSkeletonJSON(raw_dataset_path):
  """ 
  Preprocess skeletal data (json format) in the raw dataset path and 
  save an index of the contents to CSV
  
  @param raw_dataset_path File path to skeletal JSON files
  """ 
  processed_dataset_path = get_processed_dataset_path(raw_dataset_path)

  # list files in densepose (i.e., skeletal json data) folder
  files = list()
  for (dirpath, dirnames, filenames) in os.walk(raw_dataset_path):
    files += [os.path.join(dirpath, file) for file in filenames]

  # create pandas data frame with densepose data
  densepose = pd.DataFrame(dp_files, columns=["filename"])
  regex = densepose['filename'].str.extract(
    '\/(?P<dance>\w+)\/(?P<vid>[^/]+)_(?P<start_fid>[0-9]+)_(?P<relative_fid>[0-9]+)\.json', 
    flags=0, 
    expand=True)
  densepose = densepose.join(regex)

  # label dances with unique identifiers (only for the 10 original in the paper)
  dance_dict = {
      'ballet': 0, 'break': 1, 'flamenco': 2, 'foxtrot': 3, 'latin': 4,
      'quickstep': 5, 'square': 6, 'swing': 7, 'tango': 8, 'waltz': 9
  }
  densepose['dance_id'] = densepose['dance'].apply(lambda x: dance_dict.get(x))
  # drop entries with missing dance ID
  densepose.dropna(axis=0, subset=["dance_id"], inplace=True)
  # renumber index
  densepose.reset_index(drop=True, inplace=True)

  # add filepath for processed data
  densepose['processed_path'] = densepose.apply(
      lambda x: os.path.join(processed_dataset_path, x['filename'][:-5], ".npy"),
                             axis=1)
  
  # split to train, val, test, and save file indexes
  ## TO-DO: Talk to Noa about how to use `get_splits` consistently! The below code
    # only works if the same vid IDs exist between our 2 data sets!!!
  train_vids, val_vids, test_vids = get_splits(densepose['vid'].drop_duplicates())
  train = densepose[densepose['vid'].isin(train_vids)]
  val = densepose[densepose['vid'].isin(val_vids)]
  test = densepose[densepose['vid'].isin(test_vids)]
  train.to_csv("data/densepose_train_index.csv")
  val.to_csv("data/densepose_val_index.csv")
  test.to_csv("data/densepose_test_index.csv")

  ## TO-DO: transform raw data into 2-dimensional np array & save processed file
    # dimensions are: (person, body part)
  for i, fpath in enumerate(files):
    # load the json file
    with open(fpath) as f:
      skel_data = json.load(f)


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
