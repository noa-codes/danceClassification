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
    parser = argparse.ArgumentParser()

    # model specifications
    parser.add_argument("--mode", dest="mode", default='train', help="Mode is one of 'train', 'test'")
    parser.add_argument("--model", dest="model", default="baseline_lstm", help="Name of model to use")
    parser.add_argument("--encode", dest="encode", default=0, help="encode is 0 or 1, default 0")
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


def main():
    """
    Perform training of testing of many to one model
    Optionally encode your data first with a CNN
    """
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

    # Turns args into a dictionary to pass to models
    kwargs = vars(args)
    params = kwargs.copy()

    # Encode your data before using it
    if args.encode:
        print("Starting encoding...")
        make_jpg_index("/mnt/disks/disk1/raw/rgb")
        dataset = cnnDataset(args.train_path)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        encoding_model = ModelChooser("resnet18_features")
        # Run a test forward pass to save all features
        test(encoding_model, dataloader, device, save_filepath="/mnt/disks/disk1/encoded_features")

    # Load the model
    model = ModelChooser(args.model, **kwargs)
    model = model.to(device)

    # Load the encoded feature dataset
    dataset = cnnDataset(args.train_path) # TODO: make this point to the encoded features
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    val_dataset = cnnDataset(args.val_path) # TODO: make this point to the encoded features
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    if args.mode == 'train':
        print("Starting training...")
        # Save all params used to train
        json.dump(params, open(os.path.join(unique_logdir, "params.json"), 'w'), indent=2)
        # train model
        train(...)

    elif args.mode == 'test':
        print("Starting testing...")
        test(model, dataloader, device)


def test(model, dataloader, device=None, dtype=None, save_filepath=None, **kwargs):
    """
    Test your model on the dataloaded by dataloader
    """
    # Tests on batches of data from dataloader
    for (i, batch) in enumerate(dataloader):
        x, y = batch
        x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
        y = y.to(device=device, dtype=torch.long)
        scores = model(x)
        _, preds = scores.max(1)
        num_correct += (preds == y).sum()
        num_samples += preds.size(0)
        # If given a path, save the output scores
        # TODO: @Noa, this is just a janky save, can we make this save a dataset?
        if save_filepath is not None:
            np.save('{}/scores{}'.format(save_filepath, i), scores)
    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))

if __name__ == "__main__":
    main()
