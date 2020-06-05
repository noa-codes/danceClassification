import argparse
import numpy as np
import json
import os
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim import SGD, Adam, lr_scheduler
from test_tube import HyperOptArgumentParser

from utils.logger import Logger
from utils.utils import *
from model.model import *
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
        > python run.py --batch-size 100
        args.batch_size <-- 100
    """
    # parser = argparse.ArgumentParser()
    parser = HyperOptArgumentParser(strategy='random_search')

    # trainer arguments
    parser.add_argument("--gpu", dest="gpu", default='0', help="GPU number")
    parser.add_argument("--mode", dest="mode", default='train', help="Mode is one of 'train', 'test'")
    parser.add_argument("--encode", dest="encode", default=0, type=int, help="encode is 0 or 1, default 0")
    parser.add_argument("--ntrials", dest="ntrials", default=20, type=int, help="Number of trials to run for hyperparameter tuning")

    # model-specific arguments
    # (non-tunable)
    parser.add_argument("--model", dest="model", default="baseline_lstm", help="Name of model to use")
    parser.add_argument("--epochs", dest="epochs", type=int, default=10, help="Number of epochs to train for")
    # (tunable arguments)
    parser.opt_list("--batch-size", dest="batch_size", type=int, default=100, help="Size of the minibatch",
        tunable=True, options=[16, 32, 64, 128])
    parser.opt_range("--learning-rate", dest="learning_rate", type=float, default=1e-3, help="Learning rate for training",
        tunable=True, low=1e-5, high=1e-1, nb_samples=10)
    parser.opt_list("--hidden-size", dest="hidden_size", type=int, default=100, help="Dimension of hidden layers",
        tunable=True, options=[16, 32, 64, 128])
    parser.opt_list('--optimizer', dest="optimizer", type=str, default='SGD', help='Optimizer to use (default: SGD)',
        tunable=True, options=['SGD', 'Adam'])
    parser.add_argument("--patience", dest="patience", type=int, default=10, help="Learning rate decay scheduler patience, number of epochs")
    parser.opt_list('--frame-freq', dest="frame_freq", type=int, default=5, help='Frequency for sub-sampling frames from a video', tunable=False, options=[1, 5, 10, 15, 20, 25, 30, 35, 40])
    # (tcn-only arguments)
    parser.add_argument('--dropout', dest="dropout", type=float, default=0.05, help='Dropout applied to layers (default: 0.05)')
    parser.add_argument('--levels', type=int, default=8, help='# of levels for TCN (default: 8)')


    # program arguments (dataset and logger paths)
    parser.add_argument("--raw_data_path", dest="raw_data_path", default="/mnt/disks/disk1/raw", help="Path to raw dataset")
    parser.add_argument('--proc_data_path', dest="proc_data_path", default="/mnt/disks/disk1/processed", help="Path to processed dataset")
    parser.add_argument("--log", dest="log", default='', help="Unique log directory name under log/. If the name is empty, do not store logs")
    parser.add_argument("--checkpoint", dest="checkpoint", type=str, default="", help="Path to the .pth checkpoint file. Used to continue training from checkpoint")

    # create argparser
    args = parser.parse_args()
    return args

def encode_rgb(args, paths, device):
    """ Encode RGB data using a forward pass through pre-trained ResNet-18 model
    @param args Argparser object
    @param paths Dictionary of paths to raw and processed data
    @param device
    """
    print("Starting RGB encoding...")

    # initialize image Datasets and DataLoaders
    image_dataset = rawImageDataset(paths['processed']['combo']['csv']['train'])
    image_dataloader = DataLoader(image_dataset, batch_size=args.batch_size,
                                  shuffle=False, num_workers=4)
    val_image_dataset = rawImageDataset(paths['processed']['combo']['csv']['val'])
    val_image_dataloader = DataLoader(val_image_dataset, batch_size=args.batch_size,
                                      shuffle=False, num_workers=4)
    test_image_dataset = rawImageDataset(paths['processed']['combo']['csv']['test'])
    test_image_dataloader = DataLoader(test_image_dataset, batch_size=args.batch_size,
                                      shuffle=False, num_workers=4)

    # Initialize RGB CNN encoding model
    rgb_encoder = ModelChooser("resnet18_features", args)
    rgb_encoder = rgb_encoder.to(device)

    # Run a test forward pass to save all features
    print("Computing RGB CNN forward pass...")
    print("Encoding RGB training data...")
    test(rgb_encoder, image_dataloader, args, device,
         save_filepath=paths['processed']['rgb']['encode']['train'])
    print("Encoding RGB validation data...")
    test(rgb_encoder, val_image_dataloader, args, device,
         save_filepath=paths['processed']['rgb']['encode']['val'])
    print("Encoding RGB test data...")
    test(rgb_encoder, test_image_dataloader, args, device,
         save_filepath=paths['processed']['rgb']['encode']['test'])


def encode_pose(args, paths, device):
    """ Build trained encoding for pose data using PoseCNN
    @param args Argparser object
    @param paths Dictionary of data paths
    @param device
    """
    print("Starting pose encoding...")
    # Train the Densepose CNN encoding model
    pose_encoder = ModelChooser("pose_features", args)
    pose_encoder = pose_encoder.to(device)

    pose_dataset = rawPoseDataset(paths['processed']['combo']['csv']['train'])
    pose_dataloader = DataLoader(pose_dataset, batch_size=args.batch_size,
                                 shuffle=True, num_workers=4)
    val_pose_dataset = rawPoseDataset(paths['processed']['combo']['csv']['val'])
    val_pose_dataloader = DataLoader(val_pose_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=4)
    optimizer = SGD(pose_encoder.parameters(), lr=1e-3,
                          momentum=0.9, nesterov=True)

    print("Starting pose encode training...")
    train(pose_encoder, optimizer, pose_dataloader, val_pose_dataloader,
          args, device, logger)
    # point to checkpoint file -- will be used for testing
    # args.checkpoint = os.path.join(unique_logdir, "checkpoints", "best_val_loss.pth")
    print("Done with training!")

    # having trained pose_encoder, make last layer identity and encode features
    print("Starting forward pass for pose encodings...")
    pose_encoder.fcfinal = nn.Identity()
    # set shuffle to False to ensure encodings are ordered correctly
    pose_dataloader = DataLoader(pose_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=4)
    val_pose_dataloader = DataLoader(val_pose_dataset, batch_size=args.batch_size,
                                     shuffle=False, num_workers=4)
    test_pose_dataset = rawPoseDataset(paths['processed']['combo']['csv']['test'])
    test_pose_dataloader = DataLoader(test_pose_dataset, batch_size=args.batch_size,
                                     shuffle=False, num_workers=4)
    print("Encoding pose training data...")
    test(pose_encoder, pose_dataloader, args, device,
         save_filepath=paths['processed']['pose']['encode']['train'])
    print("Encoding pose validation data...")
    test(pose_encoder, val_pose_dataloader, args, device,
         save_filepath=paths['processed']['pose']['encode']['val'])
    print("Encoding pose test data...")
    test(pose_encoder, test_pose_dataloader, args, device,
         save_filepath=paths['processed']['pose']['encode']['test'])
    print("Done with encoding!")


def get_rnn_dataloaders(frame_select, batch_size, paths, shuffle=False):
    """ Construct datasets and data loaders for RNN model
    @param frame_select Range object indicating which frames to select from
        each video
    @param batch_size Batch size for DataLoaders
    @param paths Dictionary of data paths
    @param shuffle Boolean indicating whether DataLoaders should be shuffled
    @return Tuple of (train_dataloader, val_dataloader, test_dataloader)
    """
    dataset = rnnDataset(paths['processed']['rgb']['encode']['train'],
                         paths['processed']['pose']['encode']['train'],
                         paths['processed']['combo']['csv']['train'],
                         frame_select)
    val_dataset = rnnDataset(paths['processed']['rgb']['encode']['val'],
                             paths['processed']['pose']['encode']['val'],
                             paths['processed']['combo']['csv']['val'],
                             frame_select)
    test_dataset = rnnDataset(paths['processed']['rgb']['encode']['test'],
                             paths['processed']['pose']['encode']['test'],
                             paths['processed']['combo']['csv']['test'],
                             frame_select)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=shuffle, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                                shuffle=shuffle, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4)
    return dataloader, val_dataloader, test_dataloader


def get_optimizer(model, args):
    """ Construct optimizer based on args.optimizer argument
    @param model Model with parameters to use for the optimizer
    @param args Argparser object
    @return torch.optim.Optimizer objec
    """
    # generate optimizer
    if args.optimizer == "Adam":
        optimizer = Adam(model.parameters(), lr=args.learning_rate)
    if args.optimizer == "SGD":
        optimizer = SGD(model.parameters(), lr=args.learning_rate,
                                    momentum=0.9, nesterov=True)
    return optimizer


def main():
    """
    Perform training of testing of many to one model
    Optionally encode your data first with a CNN
    """
    # setup paths
    print("Setting up...")
    args = argParser()

    paths = make_paths(args.raw_data_path, args.proc_data_path)

    device = torch.device('cuda:' + args.gpu if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)

    # Set up logging
    if args.log != '':
        unique_logdir = create_unique_logdir(args.log, args.learning_rate)
        print("All logs will be saved to: ", unique_logdir)

    # create index files if they haven't been created
    if not os.path.exists(paths['processed']['combo']['csv']['train']):
        make_index(args.raw_data_path)

    # encode RGB data
    if args.encode == 1 or args.encode == 2:
        encode_rgb(args, paths, device)

    # encode pose data
    if args.encode == 1 or args.encode == 3:
        encode_pose(args, paths, device)

    # Load the temporal model
    model = ModelChooser(args.model, args)
    model = model.to(device)

    if args.mode == 'train':
        # set up (optional) Tensorboard logging
        logger = None
        if args.log != '':
            logger = Logger(unique_logdir)
            print("Will log to tensorboard: ", logger is not None)

            # save parameters used for training, only save non-function ones
            params = vars(args)
            saveable_params = {i:params[i] for i in params if not callable(params[i])}
            json.dump(saveable_params, open(os.path.join(unique_logdir, "params.json"), 'w'), indent=2, sort_keys=True)

        # load the encoded feature dataset (train and validation)
        dataloader, val_dataloader, _ = get_rnn_dataloaders(
            frame_select=range(5,305,5),
            batch_size=args.batch_size,
            paths=paths)

        print("Starting training...")
        optimizer = get_optimizer(model, args)
        train(model, optimizer, dataloader, val_dataloader, args, device, logger)

    elif args.mode == 'test':
        print("Starting testing...")
        # load the encoded feature dataset (train and validation)
        _, _, test_dataloader = get_rnn_dataloaders(
            frame_select=range(args.frame_freq, 300 + args.frame_freq, args.frame_freq),
            batch_size=args.batch_size,
            paths=paths)

        acc, loss = test(model, test_dataloader, args, device)
        print(f'Test Loss: {loss} | Test Accuracy: {acc}')

    # hyperparameter tuning
    elif args.mode == 'tune':
        print("Starting tuning...")
        results = []
        best_val_loss = np.inf
        # loop over trials
        for i, trial in enumerate(args.trials(args.ntrials)):
            print(f'Running experiment {i} out of {args.ntrials}...')
            val_loss = tune(trial, paths, device)
            params = vars(trial)

            # compare to current best trial
            if val_loss < best_val_loss:
                print(f"Achieved new minimum validation loss: {val_loss}")
                best_val_loss = val_loss
                json.dump(params, open(os.path.join(unique_logdir, "params.json"), 'w'), indent=2, sort_keys=True)

            # store experiment and result
            params["val_loss"] = val_loss
            results.append(params)

        # save results to data frame
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(unique_logdir, "tuning_params.csv"))
        print(f"Best validation loss: {best_val_loss}")


def tune(trial, paths, device):
    """ Run one trial of hyperparameter tuning
    @param trial Labeled tuple of parameters for a given trial
    @param paths Dictionary of data paths, used to construct dataloaders
    @param device Pytorch device
    @return Validation loss from the trial
    """
    # generate the model
    model = ModelChooser(trial.model, trial)
    model = model.to(device)

    # generate data loaders
    dataloader, val_dataloader, _ = get_rnn_dataloaders(
        frame_select=range(trial.frame_freq, 300 + trial.frame_freq, trial.frame_freq),
        batch_size=trial.batch_size,
        paths=paths)

    # generate optimizer
    optimizer = get_optimizer(model, trial)

    # train and return validation loss for this trial
    val_loss = train(model, optimizer, dataloader, val_dataloader, trial, device)
    return val_loss


def train(model, optimizer, dataloader, val_dataloader, args, device, logger=None):
    # extract arguments
    learning_rate = args.learning_rate
    epochs = args.epochs
    patience = args.patience

    # set up logging
    save_to_log = logger is not None
    logdir = logger.get_logdir() if logger is not None else None

    # loss criterion
    criterion = nn.CrossEntropyLoss()
    # record minimum validation loss
    min_val_loss = None
    
    # set up early stopping
    early_stopping_counter = 0
    # Limit step to wait for 2x patience.
    early_stopping_limit = 2 * patience

    # set up scheduler for learning rate decay
    # we can make the factor into a tunable parameter if needed
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.5, patience=patience)

    for e in range(epochs):
        # initialize loss
        epoch_loss = []
        num_correct = 0
        num_samples = 0
        model.train()

        # train for one epoch
        for t, (x,y) in enumerate(tqdm(dataloader)):
            x = x.to(device=device, dtype=torch.float)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            loss = criterion(scores, y)
            # need to zero out gradients between batches
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
            # calculate accuracy
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)

        # End of epoch, run validations
        model.eval()
        with torch.no_grad():
            epoch_train_loss = np.mean(epoch_loss)
            epoch_train_acc = float(num_correct) / num_samples
            epoch_val_acc, epoch_val_loss = test(model, val_dataloader, args, device)
            scheduler.step(epoch_val_loss)

            # Check for early stopping
            if min_val_loss is None or epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            if early_stopping_counter >= early_stopping_limit:
                print("Early stopping after waiting {} epochs".format(early_stopping_limit))
                break

            # Add to logger on tensorboard at the end of an epoch
            if save_to_log:
                logger.scalar_summary("epoch_train_loss", epoch_train_loss, e)
                logger.scalar_summary("epoch_train_acc", epoch_train_acc, e)
                logger.scalar_summary("epoch_val_loss", epoch_val_loss, e)
                logger.scalar_summary("epoch_val_acc", epoch_val_acc, e)

                # Save epoch checkpoint
                if e % 10 == 0:
                    save_checkpoint(logdir, model, optimizer, e, epoch_train_loss,
                                   learning_rate)
                # Save best validation checkpoint
                if epoch_val_loss == min_val_loss:
                    save_checkpoint(logdir, model, optimizer, e, epoch_train_loss,
                                    learning_rate, best="val_loss")

            print('Epoch {} | train loss: {:.3f} | val loss: {:.3f} | train acc: {:.3f} | val acc: {:.3f}'
                .format(e + 1, epoch_train_loss, epoch_val_loss, epoch_train_acc,
                        epoch_val_acc))

    # return the best validation loss
    return min_val_loss


def test(model, dataloader, args, device, save_filepath=None):
    """
    Test your model on the dataloaded by dataloader
    """
    # load model from checkpoint
    if args.checkpoint:
        model = load_checkpoint(args.checkpoint, model, device)

    criterion = nn.CrossEntropyLoss()

    aggregate_loss = []
    all_scores = []
    num_correct = 0
    num_samples = 0
    pred_y = []
    true_y = []

    # Tests on batches of data from dataloader
    model.eval()
    with torch.no_grad():
        for (i, batch) in enumerate(tqdm(dataloader)):
            x, y = batch
            x = x.to(device=device, dtype=torch.float)
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            loss = criterion(scores, y)
            aggregate_loss.append(loss.item())
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)

            # Record scores to save
            if save_filepath is not None:
                all_scores.append(scores)

            # Record the predicted and true classes
            true_y.append(y)
            pred_y.append(preds)

    if save_filepath:
        # convert torch to CPU and then to NumPy
        encoding = torch.cat(all_scores).cpu().numpy()
        # save as NumPy file
        np.save(save_filepath, encoding)

    # save the predicted and true classes in a NumPy file
    true_y = torch.cat(true_y).cpu().numpy()
    pred_y = torch.cat(pred_y).cpu().numpy()
    np.save(save_filepath, true_y)
    np.save(save_filepath, pred_y)

    # Report accuracy and average loss
    acc = float(num_correct) / num_samples

    # Calculate average loss
    average_loss = np.mean(aggregate_loss)

    return acc, average_loss


if __name__ == "__main__":
    main()
