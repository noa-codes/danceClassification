import argparse
import numpy as np
import json
import os
import torch
import torch.nn as nn

from utils import *
from model import *
from dataset import *

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

def argParser():
    """
    This function creates a parser object which parses all the flags from the command line
    We can access the parsed command line values using the args object returned by this function
    Usage:
        First field is the flag name.
        dest=NAME is the name to reference when using the parameter (args.NAME)
        default is the default value of the parameter
    Example:
        > python run.py --gpu 0
        args.gpu <-- 0
    """
    parser = argparse.ArgumentParser()

    # model specifications
    parser.add_argument("--mode", dest="mode", default='train', help="Mode is one of 'train', 'test'")
    parser.add_argument("--model", dest="model", default="baseline_lstm", help="Name of model to use")
    parser.add_argument("--preprocess", dest="preprocess", default=0, help="Preprocess is 0 or 1, default 0")
    parser.add_argument("--gpu", dest="gpu", type=str, default='0', help="The gpu number if there's more than one gpu")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=100, help="Size of the minibatch")
    
    # dataset and logger paths
    parser.add_argument("--train-path", dest="train_path", help="Training data file")
    parser.add_argument("--val-path", dest="val_path", help="Validation data file")
    parser.add_argument("--log", dest="log", default='', help="Unique log directory name under log/. If the name is empty, do not store logs")
    
    # create argparser
    args = parser.parse_args()
    return args


def train():
    pass


def test():
    pass

        

def main():
    # setup
    print("Setting up...")
    args = argParser()
    device = torch.device('cuda:' + args.gpu if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)
    
#  Set up logging ----------------------------------------------
#     unique_logdir = create_unique_logdir(args.log, args.lr)
#     logger = Logger(unique_logdir) if args.log != '' else None
#     print("All training logs will be saved to: ", unique_logdir)
#     print("Will log to tensorboard: ", logger is not None)

    # Prepare for forward pass on CNN
    if args.preprocess:
 
        print("Indexing JPG files...")
        make_jpg_index("/mnt/disks/disk1/raw/rgb")
        
        # build Dataset and DataLoader objects
        print("Creating Datasets...")
        train_dataset = cnnDataset(args.train_path)
        val_dataset = cnnDataset(args.val_path)

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Many-to-one or RNN model
    else: 
        pass
    

    # Turns args into a dictionary to pass to models
    kwargs = vars(args) 
    params = kwargs.copy()

    print("Done!")

    # build model
    model = ModelChooser(args.model, **kwargs)
    model = model.to(device)

    if args.mode == 'train':
        print("Starting training...")
        # Save all params used to train
        json.dump(params, open(os.path.join(unique_logdir, "params.json"), 'w'), indent=2)
        # train model
        train(...)

    elif args.mode == 'test':
        print("Starting testing...")
        test(...)


if __name__ == "__main__":
    main()
