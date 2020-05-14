import argparse
import numpy as np
import json
import os
import torch
import torch.nn as nn
import torchvision.models as models

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
    pass
    parser = argparse.ArgumentParser()

    ### ADD ARGUMENTS, for example:
    ### parser.add_argument("--mode", dest="mode", default='train', help="Mode is one of 'train', 'test', 'generate'")

    args = parser.parse_args()
    return args


def train():
    pass


def main():
    # setup
    print("Setting up...")
    args = argParser()
    args.is_stream = True if args.is_stream == 1 else False
    device = torch.device('cuda:' + args.gpu if torch.cuda.is_available() else "cpu")
    unique_logdir = create_unique_logdir(args.log, args.lr)
    logger = Logger(unique_logdir) if args.log != '' else None
    print("Using device: ", device)
    print("All training logs will be saved to: ", unique_logdir)
    print("Will log to tensorboard: ", logger is not None)

    # build dataset object
    print("Creating Dataset...")

    ###### TO DO #######
    # BUILD DATASETS  ##
    ####################

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
        train()

    elif args.mode == 'test':
        print("Starting testing...")
        test(...)



def test(model, dataloader, device=None, dtype=None, save_scores=None, **kwargs):
    """
    Loop over batches in train_dataloader and train
    """
    # Tests on batches of data from dataloader
    for batch in dataloader:
        x, y = batch
        x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
        y = y.to(device=device, dtype=torch.long)
        scores = model(x)
        _, preds = scores.max(1)
        num_correct += (preds == y).sum()
        num_samples += preds.size(0)
    acc = float(num_correct) / num_samples
    # if save_scores:
        #


if __name__ == "__main__":
    main()
