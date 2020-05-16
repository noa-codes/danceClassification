from dataset import *
import sys
from tqdm import tqdm

raw_dataset_path = sys.argv[1]

preprocessSkeletonJSON(raw_dataset_path)