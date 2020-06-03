import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import json
from tqdm import tqdm

C_TRAIN_CSV = "train_index.csv"
C_VAL_CSV = "val_index.csv"
C_TEST_CSV = "test_index.csv"
C_CSV_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

dance_dict = {
      'ballet': 0, 'break': 1, 'flamenco': 2, 'foxtrot': 3, 'latin': 4,
      'quickstep': 5, 'square': 6, 'swing': 7, 'tango': 8, 'waltz': 9
  }

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


def make_index(raw_dataset_path):
  """
  Create an index of all JPG files at raw_dataset_path, and save to CSV

  @param raw_dataset_path File path to JPG files
  """
  # location to save processed data
  processed_dataset_path = get_processed_dataset_path(raw_dataset_path)

  # list files in folder
  rgb_files = list()
  for (dirpath, dirnames, filenames) in os.walk(os.path.join(raw_dataset_path, "rgb")):
    rgb_files += [os.path.join(dirpath, file) for file in filenames]

  # create pandas data frame with RGB data
  df = pd.DataFrame(rgb_files, columns=["rgb_filename"])
  # extract vid, start_fid, relative_fid from filename
  regex = df['rgb_filename'].str.extract(
    '\/(?P<dance>\w+)\/(?P<vid>[^/]+)_(?P<start_fid>[0-9]+)_(?P<relative_fid>[0-9]+)\.jpg',
    flags=0,
    expand=True)
  df = df.join(regex).dropna(axis=0, subset=["dance"])

  # add path for json files
  df['pose_filename'] = df['rgb_filename'].map(
      lambda x: x.replace(".jpg", ".json").replace("rgb", "densepose"))

  df['dance_id'] = df['dance'].apply(lambda x: dance_dict.get(x))
  df.dropna(axis=0, subset=["dance_id"], inplace=True)
  df.reset_index(drop=True, inplace=True)

  # split video IDs to train, val, test
  train_vids, val_vids, test_vids = get_splits(df['vid'].drop_duplicates())
  # subset index file
  train = df[df['vid'].isin(train_vids)].reset_index(drop=True)
  val = df[df['vid'].isin(val_vids)].reset_index(drop=True)
  test = df[df['vid'].isin(test_vids)].reset_index(drop=True)
  # save CSV indexes
  train.to_csv(os.path.join(C_CSV_DIR, C_TRAIN_CSV))
  val.to_csv(os.path.join(C_CSV_DIR, C_VAL_CSV))
  test.to_csv(os.path.join(C_CSV_DIR, C_TEST_CSV))

  return train, val, test


def pad_skeletons(json_file, skeleton_array):
  """
  Zero pad frames containing less than 20 PoseNet skeletons.

  @param json_file Skeleton data for a frame in JSON format
  @param skeleton_array Numpy array containing the PoseNet skeletons for a frame.
  """
  skeleton_pad = np.zeros((20, 17, 2))
  # zero pad any skeleton arrays with less than 20 skeletons
  if len(json_file) > 0:
    skeleton_pad[:skeleton_array.shape[0], :skeleton_array.shape[1], :skeleton_array.shape[2]] \
        = skeleton_array

  return skeleton_pad


def standardize_skeletons(skeleton_array):
  """
  Center and normalize the (x,y) coordinates of each PoseNet skeleton by
  constructing a bounding box around each skeleton.

  @param skeleton_array Numpy array of dim (17, 2, 20) containing the PoseNet
                        skeletons for a frame.
  """
  # get min and max (x,y) coordinates for each skeleton to use as bounding box
  xy_min = np.amin(skeleton_array, axis=0) # dim is (2, 20)
  xy_max = np.amax(skeleton_array, axis=0) # dim is (2, 20)
  # compute the center of the bounding box
  dist_to_center = (xy_max - xy_min) / 2
  xy_center = xy_min + dist_to_center
  # center and normalize each skeleton w.r.t. the center of its bounding box
  std_skeleton = np.divide((skeleton_array - xy_center), dist_to_center, \
                          out = np.zeros_like(skeleton_array - xy_center), \
                          where = dist_to_center!=0)

  return std_skeleton


def preprocessSkeletonJSON(processed_dataset_path):
  """
  Preprocess skeletal data (json format) and save the results in
  the `processed/densepose` directory.

  @param processed_dataset_path File path to the processed data sets
  """
  # read in the index files specifying the train, val, and test split
  train = pd.read_csv(os.path.join(C_CSV_DIR, C_TRAIN_CSV), \
                    index_col=0)
  val = pd.read_csv(os.path.join(C_CSV_DIR, C_VAL_CSV), \
                    index_col=0)
  test = pd.read_csv(os.path.join(C_CSV_DIR, C_TEST_CSV), \
                    index_col=0)
  # create new column containing filepath to processed skeleton data
  files = [train, val, test]
  for f in files:
    f["processed_path"] = f["pose_filename"].apply( \
                    lambda x: get_processed_dataset_path(x). \
                    replace(".json", ".npy"))

  # create subdirectories within the processed folder
  for dance in dance_dict.keys():
    os.mkdir(os.path.join(processed_dataset_path, dance))

  # transform json files into 3D numpy arrays
  # output dimensions: (num_body_parts, coordinates, max_people) or (17, 2, 20)
  for file in files:
    for i, fpath in enumerate(tqdm(file['pose_filename'])):
      with open(fpath) as f:
        # load json file
        json_file = json.load(f)

        # remove unnecessary fields
        for i in range(len(json_file)):
          # remove first entry, which is a person index (e.g., `person0`)
          json_file[i].pop(0)
          for j in range(len(json_file[i])):
            # remove the first entry, which is a body part label (e.g., `nose`)
            json_file[i][j] = json_file[i][j][1]

        # convert nested lists into a numpy array
        np_file = np.asarray(json_file)
        # zero pad frames with less than 20 skeletons
        np_file_pad = pad_skeletons(json_file, np_file)
        # switch ordering of axes to dimensions: (num_body_parts, coordinates, num_people)
        np_file_pad = np.transpose(np_file_pad, axes=(1,2,0))

        # center and normalize each skeleton w.r.t. the center of its bounding box
        np_file_pad = standardize_skeletons(np_file_pad)

        # get class
        y = file.loc[file['pose_filename'] == fpath, 'dance_id'].squeeze()
        # save to file
        obs = np.asarray([np_file_pad, y])
        out_path = file.loc[file['pose_filename'] == fpath, 'processed_path'].squeeze()
        np.save(out_path, obs, allow_pickle=True)


class rawImageDataset(Dataset):
    """ Custom dataset for CNN image data
    index_filepath: path to an index
    """
    def __init__(self, index_filepath):
      # load file index
      self.file_index =  pd.read_csv(index_filepath, index_col=0)

      # create a transform
      self.transform = transforms.Compose([
          transforms.Resize([256, 256]),
          transforms.ToTensor(),
          transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])])

    def __len__(self):
      """ Return number of obs in the dataset
      """
      return len(self.file_index)

    def __getitem__(self, index):
      """ Return X, y for a single observation
      """
      # get filepath
      path = self.file_index["rgb_filename"].iloc[index]

      # load and transform image
      X = Image.open(path)
      X = self.transform(X)

      # get class
      y = self.file_index["dance_id"][index]

      return X, y

    def get_X(self, path):
        """
        Return just the input data X, resize to 256
        """
        X = Image.open(path)
        X = self.transform(X)
        return X


class rawPoseDataset(Dataset):
    """ Custom dataset for CNN PoseNet (i.e., skeleton) data
    """
    def __init__(self, index_filepath):
      # load file index
      self.file_index =  pd.read_csv(index_filepath, index_col=0)
      # create new column containing filepath to processed skeleton data
      self.file_index["processed_path"] = self.file_index["pose_filename"].apply( \
                        lambda x: get_processed_dataset_path(x). \
                        replace(".json", ".npy"))


    def __len__(self):
      """ Return number of obs in the dataset
      """
      return len(self.file_index)

    def __getitem__(self, index):
      """ Return X, y for a single observation
      """
      # get filepath
      path = self.file_index["processed_path"].iloc[index]

      # load the processed skeleton data
      file = np.load(path, allow_pickle = True)
      # extract 3D numpy array containing skeletons
      # Rearrange from (17, 2, 20) to (20, 17, 2)
      X = np.transpose(file[0], [ 2, 0, 1])
      # get class
      y = file[1]

      return X, y

    def get_X(self, path):
        """
        Return just the input data X with transforms
        """
        file = np.load(path, allow_pick = True)
        X = np.transpose(file[0], [ 2, 0, 1])
        return X


class rnnDataset(Dataset):
    """ Custom dataset for RNN image data

    """
    def __init__(self, rgb_encode_path, pose_encode_path,
      index_path, selection):
        # get maximum sequence length
        self.seq_len = len(selection)

        # load encoded data
        self.rgb_encodings = np.load(rgb_encode_path)
        self.pose_encodings = np.load(pose_encode_path)

        # filter index to desired frames
        index = pd.read_csv(index_path, index_col=0)
        subsample = index[index['relative_fid'].isin(selection)]

        # reshape to get list of frames for each unique video
        subsample = subsample[['vid', 'start_fid', 'relative_fid', 'dance_id']] \
            .reset_index() \
            .sort_values(by=["vid", "start_fid", "relative_fid"]) \
            .groupby(['vid', 'start_fid', 'dance_id'])['index'] \
            .apply(list) \
            .reset_index(name='fids')

        # save reshaped index
        self.file_index = subsample

    def __len__(self):
      """ Return number of obs in the dataset
      """
      return len(self.file_index)

    def __getitem__(self, index):
      """ Return X, y for a single observation
      """
      # get frame IDs
      fids = self.file_index["fids"].iloc[index]
      n_frames = len(fids)
      # extract encodings corresponding to frame IDs
      # x1 has dimension (num_frames, rgb_encoding_dim)
      # x2 has dimension (num_frames, pose_encoding_dim)
      x1 = self.rgb_encodings[fids]
      x2 = self.pose_encodings[fids]

      # concatenate encodings
      # X has dimension (num_frames, rgb_encoding_dim + pose_encoding_dim)
      X = np.concatenate((x1, x2), axis=1)

      # pad to sequence length (with zeros)
      if n_frames < self.seq_len:
            X = np.pad(X, ((0, self.seq_len - n_frames),(0,0)),
                       mode='constant', constant_values=(0))
      # get class
      y = self.file_index["dance_id"][index]

      return X, y



