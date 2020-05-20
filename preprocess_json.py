import os
import json
import collections
import pprint
from dataset import *
import sys
from tqdm import tqdm

pp = pprint.PrettyPrinter(depth=6)

# specify relevant data paths
posedir = '/mnt/disks/disk1/raw/densepose/'
rgbdir = '/mnt/disks/disk1/raw/rgb'
posedir_processed = '/mnt/disks/disk1/processed/densepose'

# specify the 10 dance types that we are focusing on
dance_dict = {
      'ballet': 0, 'break': 1, 'flamenco': 2, 'foxtrot': 3, 'latin': 4,
      'quickstep': 5, 'square': 6, 'swing': 7, 'tango': 8, 'waltz': 9
}

# function that creates an empty json file
def write_empty_json(filename):
    with open(filename, 'w') as f:
        f.write('[]\n')

# add empty json files for all the frames that exist in the raw rgb data but
    # don't exist in the raw skeletal data
pose_counts = {}
rgb_counts = {}
missing = 0
for d in dance_dict:
    poseListing = os.listdir(os.path.join(posedir, d))
    poseListing = [os.path.splitext(x)[0] for x in poseListing]
    pose_counts[d] = len(poseListing)
    
    rgbListing = os.listdir(os.path.join(rgbdir, d))
    rgbListing = [os.path.splitext(x)[0] for x in rgbListing]
    rgb_counts[d] = len(rgbListing)
    
    # figure out which ones are missing
    if len(poseListing) != len(rgbListing):
        for f in rgbListing:
            if f not in poseListing:
                missing += 1
                filename = os.path.join(posedir, d, f) + '.json'
                write_empty_json(filename)
                print('Adding {}'.format(filename))

# print statistics on the mismatch between the two datasets
print('missing: {}'.format(missing))
dirListing = os.listdir(posedir_processed)
print("Total pose: {}".format(sum(pose_counts.values())))
pp.pprint(pose_counts)
print("Total pose_proc: {}".format(len(dirListing)))
print("Total rgb: {}".format(sum(rgb_counts.values())))
pp.pprint(rgb_counts)

for d in dance_dict:
    if pose_counts[d] != rgb_counts[d]:
        print('Mismatch on {}, rgb: {}, pose: {}'.format(d, rgb_counts[d], pose_counts[d]))


# process all the raw json files stored at the path
preprocessSkeletonJSON(posedir)