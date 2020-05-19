import argparse
import numpy as np
import json
import os
from datetime import *
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim

from logger import Logger
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
        > python run.py --batch-size 100
        args.batch_size <-- 100
    """
    parser = argparse.ArgumentParser()

    # model specifications
    parser.add_argument("--gpu", dest="gpu", default='0', help="GPU number")
    parser.add_argument("--mode", dest="mode", default='train', help="Mode is one of 'train', 'test'")
    parser.add_argument("--model", dest="model", default="baseline_lstm", help="Name of model to use")
    parser.add_argument("--encode", dest="encode", default=0, type=int, help="encode is 0 or 1, default 0")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=100, help="Size of the minibatch")
    parser.add_argument("--learning-rate", dest="learning_rate", type=float, default=1e-3, help="Learning rate for training")
    parser.add_argument("--epochs", dest="epochs", type=int, default=10, help="Number of epochs to train for")

    # dataset and logger paths
    parser.add_argument("--raw_data_path", dest="raw_data_path", default="/mnt/disks/disk1/raw", help="Path to raw dataset")
    parser.add_argument('--proc_data_path', dest="proc_data_path", default="/mnt/disks/disk1/processed", help="Path to processed dataset")
    parser.add_argument("--log", dest="log", default='', help="Unique log directory name under log/. If the name is empty, do not store logs")
    parser.add_argument("--models_path", dest="models_path",
                        default="/mnt/disks/disk1/models", help="Path to models")

    # create argparser
    args = parser.parse_args()
    return args


def main():
    """
    Perform training of testing of many to one model
    Optionally encode your data first with a CNN
    """
    # setup paths
    print("Setting up...")
    args = argParser()

    # Dictionary to paths nesting as follows:
    paths = {'raw': {'rgb': '', 'pose': ''},
             'processed': {
                'rgb': {
                    'csv': {'train': '', 'val': '', 'test': ''},
                    'encode': {'train': '', 'val': '', 'test': ''}},
                'pose': {
                    'csv': {'train': '', 'val': '', 'test': ''},
                    'encode': {'train': '', 'val': '', 'test': ''},
                'combo' : ''}}
    }
    paths['raw']['rgb'] = os.path.join(args.raw_data_path, 'rgb')
    paths['raw']['pose'] = os.path.join(args.raw_data_path, 'densepose')
    paths['processed']['rgb']['csv']['train'] = os.path.join(args.proc_data_path, 
                                                             'rgb', C_RGB_TRAIN_CSV)
    paths['processed']['rgb']['csv']['val'] = os.path.join(args.proc_data_path, 
                                                           'rgb', C_RGB_VAL_CSV)
    paths['processed']['rgb']['csv']['test'] = os.path.join(args.proc_data_path, 
                                                            'rgb', C_RGB_TEST_CSV)
    paths['processed']['rgb']['encode']['train'] = os.path.join(args.proc_data_path,
                                                            "rgb/encoded_features_train.pt")
    paths['processed']['rgb']['encode']['val'] = os.path.join(args.proc_data_path,
                                                                  "rgb/encoded_features_val.pt")
    paths['processed']['rgb']['encode']['test'] = os.path.join(args.proc_data_path, "rgb/encoded_features_test.pt")
    paths['processed']['pose']['csv']['train'] = os.path.join(args.proc_data_path, 
                                                              'densepose', C_POSE_TRAIN_CSV)
    paths['processed']['pose']['csv']['val'] = os.path.join(args.proc_data_path, 
                                                            'densepose', C_POSE_VAL_CSV)
    paths['processed']['pose']['csv']['test'] = os.path.join(args.proc_data_path, 
                                                             'densepose', C_POSE_TEST_CSV)
    paths['processed']['pose']['encode']['train'] = os.path.join(args.proc_data_path, "densepose/encoded_features_train.pt")
    paths['processed']['pose']['encode']['val'] = os.path.join(args.proc_data_path, "densepose/encoded_features_val.pt")
    paths['processed']['pose']['encode']['test'] = os.path.join(args.proc_data_path, "densepose/encoded_features_test.pt")
    paths['processed']['combo'] = os.path.join(args.proc_data_path, "combo")

    device = torch.device('cuda:' + args.gpu if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)

    # Set up logging
    unique_logdir = create_unique_logdir(args.log, args.learning_rate)
    logger = Logger(unique_logdir) if args.log != '' else None
    print("All training logs will be saved to: ", unique_logdir)
    print("Will log to tensorboard: ", logger is not None)

    # Turns args into a dictionary to pass to models
    kwargs = vars(args)
    params = kwargs.copy()
    
    if args.encode == 1 or args.encode == 2:
        print("Starting rgb encoding...")

        ####################################
        # Encode RGB data
        ####################################
        # Raw RGB data hasn't been indexed, so index it
        if not os.path.exists(paths['processed']['rgb']['csv']['train']):
            make_jpg_index(paths['raw']['rgb'])

        # initialize image Datasets and DataLoaders
        image_dataset = rawImageDataset(paths['processed']['rgb']['csv']['train'])
        image_dataloader = DataLoader(image_dataset, batch_size=args.batch_size, 
                                      shuffle=False, num_workers=4)
        val_image_dataset = rawImageDataset(paths['processed']['rgb']['csv']['val'])
        val_image_dataloader = DataLoader(val_image_dataset, batch_size=args.batch_size, 
                                          shuffle=False, num_workers=4)

        # Forward pass through the RGB CNN encoding model
        rgb_encoder = ModelChooser("resnet18_features")
        rgb_encoder = rgb_encoder.to(device)
        
        # Run a test forward pass to save all features
        print("Computing RGB CNN forward pass...")
        test(rgb_encoder, image_dataloader, device, 
             save_filepath=paths['processed']['rgb']['encode']['train'])
        test(rgb_encoder, val_image_dataloader, device, 
             save_filepath=paths['processed']['rgb']['encode']['val'])

    if args.encode == 1 or args.encode == 3:
        print("Starting pose encoding...")
        ####################################
        # Encode pose data
        ####################################
        # Train the Densepose CNN encoding model
        pose_encoder = ModelChooser("pose_features")
        pose_encoder = pose_encoder.to(device)
        
        pose_dataset = rawPoseDataset(paths['processed']['pose']['csv']['train'])
        pose_dataloader = DataLoader(pose_dataset, batch_size=args.batch_size, 
                                     shuffle=False, num_workers=4)
        val_pose_dataset = rawPoseDataset(paths['processed']['pose']['csv']['val'])
        val_pose_dataloader = DataLoader(val_pose_dataset, batch_size=args.batch_size, 
                                         shuffle=False, num_workers=4)
        optimizer = torch.optim.SGD(pose_encoder.parameters(), lr=.01,
                              momentum=0.9, nesterov=True)
        
        print("Starting pose encode training...")
        train(pose_encoder, optimizer, pose_dataloader, val_pose_dataloader,
              device, args.epochs)
        t = datetime.utcnow()
        filename = 'pose_encoder_{:02d}-{:02d}_{:02d}_{:02d}_{:02d}'.format(
            t.month, t.day, t.hour, t.minute, t.second)
        torch.save(pose_encoder, os.path.join(args.model_path, filename))
        
        # having trained pose_encoder, make laster layer identity
        pose_encoder.fcfinal = nn.Identity()
        test(pose_encoder, pose_dataloader, device, 
             save_filepath=paths['processed']['pose']['encode']['train'])
        test(pose_encoder, val_pose_dataloader, device, 
             save_filepath=paths['processed']['pose']['encode']['val'])

        ####################################
        # TODO: Concatenate feature data! into paths['processed']['combo']
        ####################################


    # Load the model
    model = ModelChooser(args.model, **kwargs)
    model = model.to(device)

    # Load the encoded feature dataset
    frame_select = range(0,300,5)
    # TODO make this use combined features
    dataset = rnnDataset(paths['processed']['pose']['encode']['train'], paths['processed']['pose']['csv']['train'], frame_select)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    val_dataset = rnnDataset(paths['processed']['pose']['encode']['val'],  paths['processed']['pose']['csv']['val'], frame_select)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    if args.mode == 'train':
        print("Starting training...")
        # Save all params used to train
        json.dump(params, open(os.path.join(unique_logdir, "params.json"), 'w'), indent=2)

        # TODO: Better way to pick the optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,
                     momentum=0.9, nesterov=True)
        # train model
        train(model, optimizer, dataloader, val_dataloader, device, args.epochs, 
              logger=logger, **kwargs)

    elif args.mode == 'test':
        print("Starting testing...")
        test(model, dataloader, device)

def train(model, optimizer, dataloader, val_dataloader, device, epochs=10, dtype=None,
          logger=None, **kwargs):

    criterion = nn.CrossEntropyLoss()

    save_to_log = logger is not None
    logdir = logger.get_logdir() if logger is not None else None

    for e in range(epochs):
        # initialize loss
        epoch_loss = []
        num_correct = 0
        num_samples = 0
        model.train()

        # train for one epoch
        for t, (x,y) in enumerate(tqdm(dataloader)):
            model.train()
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            loss = criterion(scores, y)
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
            epoch_val_acc, epoch_val_loss = \
                test(model, optimizer, val_dataloader, device)

        # Add to logger on tensorboard at the end of an epoch
        if save_to_log:
            logger.scalar_summary("epoch_train_loss", epoch_train_loss, e)
            logger.scalar_summary("epoch_train_acc", epoch_train_acc, e)
            logger.scalar_summary("epoch_val_loss", epoch_val_loss, e)
            logger.scalar_summary("epoch_val_acc", epoch_val_acc, e)

            # TO DO: Save epoch checkpoint
            # if epoch % log_every == 0:
            #     save_checkpoint(logdir, model, optimizer, epoch, epoch_average_loss, lr)
            # # Save best validation checkpoint
            # if epoch_val_loss == min_val_loss:
            #     save_checkpoint(logdir, model, optimizer, epoch, epoch_average_loss, lr, "val_ppl")

        print('Epoch {} | train loss: {} | val loss: {} | train acc: {} | val acc: {}' \
            .format(e + 1, epoch_train_loss, epoch_val_loss, epoch_train_acc, epoch_val_acc))


def test(model, dataloader, device, dtype=None, save_filepath=None, **kwargs):
    """
    Test your model on the dataloaded by dataloader
    """

    criterion = nn.CrossEntropyLoss()

    aggregate_loss = []
    all_scores = []
    num_correct = 0
    num_samples = 0

    # Tests on batches of data from dataloader
    model.eval()
    with torch.no_grad():
        for (i, batch) in enumerate(tqdm(dataloader)):
            x, y = batch
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
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

    if save_filepath:
        encoding = torch.cat(all_scores)
        torch.save(encoding, save_filepath)

    # Report accuracy and average loss
    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))

    # Calculate average loss
    average_loss = np.mean(aggregate_loss)

    return acc, average_loss


if __name__ == "__main__":
    main()
