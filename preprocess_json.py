from dataset import *
import sys
from tqdm import tqdm

# get the path to the unprocessed files, which is fed in as a command line arg
raw_dataset_path = sys.argv[1]

# process all the raw json files stored at the path
preprocessSkeletonJSON(raw_dataset_path)