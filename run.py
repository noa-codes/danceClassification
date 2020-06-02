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
from timeit import default_timer as timer


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

    # trainer arguments
    parser.add_argument("--gpu", dest="gpu", default='0', help="GPU number")
    parser.add_argument("--mode", dest="mode", default='train', help="Mode is one of 'train', 'test'")
    parser.add_argument("--encode", dest="encode", default=0, type=int, help="encode is 0 or 1, default 0")

    # model-specific arguments
    parser.add_argument("--model", dest="model", default="baseline_lstm", help="Name of model to use")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=100, help="Size of the minibatch")
    parser.add_argument("--learning-rate", dest="learning_rate", type=float, default=1e-3, help="Learning rate for training")
    parser.add_argument("--epochs", dest="epochs", type=int, default=10, help="Number of epochs to train for")
    parser.add_argument("--hidden-size", dest="hidden_size", type=int, default=100, help="Dimension of hidden layers")
    parser.add_argument('--dropout', dest="dropout", type=float, default=0.05, help='Dropout applied to layers (default: 0.05)')
    parser.add_argument('--levels', type=int, default=8, help='# of levels for TCN (default: 8)')
    parser.add_argument('--optim', dest="optimizer", type=str, default='SGD', help='Optimizer to use (default: SGD)')
    parser.add_argument("--patience", dest="patience", type=int, default=10, help="Learning rate decay scheduler patience, number of epochs")
    
    # program arguments (dataset and logger paths)
    parser.add_argument("--raw_data_path", dest="raw_data_path", default="/mnt/disks/disk1/raw", help="Path to raw dataset")
    parser.add_argument('--proc_data_path', dest="proc_data_path", default="/mnt/disks/disk1/processed", help="Path to processed dataset")
    parser.add_argument("--log", dest="log", default='', help="Unique log directory name under log/. If the name is empty, do not store logs")
    parser.add_argument("--checkpoint", dest="checkpoint", type=str, default="", help="Path to the .pth checkpoint file. Used to continue training from checkpoint")

    # create argparser
    args = parser.parse_args()
    return args

def encode_rgb(args, paths, device):
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
    optimizer = torch.optim.SGD(pose_encoder.parameters(), lr=1e-3,
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
    unique_logdir = create_unique_logdir(args.log, args.learning_rate)
    logger = Logger(unique_logdir) if args.log != '' else None
    print("All training logs will be saved to: ", unique_logdir)
    print("Will log to tensorboard: ", logger is not None)

    # Turns args into a dictionary to pass to models
    kwargs = vars(args)
    params = kwargs.copy()
    # Save all params used to train
    if logger:
        json.dump(params, open(os.path.join(unique_logdir, "params.json"), 'w'), indent=2)

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

    # Load the encoded feature dataset (train and validation)
    frame_select = range(5,305,5)
    
    dataset = rnnDataset(paths['processed']['rgb']['encode']['train'], 
                         paths['processed']['pose']['encode']['train'], 
                         paths['processed']['combo']['csv']['train'], 
                         frame_select)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, 
                            shuffle=False, num_workers=4)
    val_dataset = rnnDataset(paths['processed']['rgb']['encode']['val'],  
                             paths['processed']['pose']['encode']['val'], 
                             paths['processed']['combo']['csv']['val'], 
                             frame_select)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, 
                                shuffle=False, num_workers=4)
    
    if args.mode == 'train':
        print("Starting training...")
        if args.optimizer == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        if args.optimizer == "SGD":   
            optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,
                                        momentum=0.9, nesterov=True)
        
        train(model, optimizer, dataloader, val_dataloader, args, device, logger)

    elif args.mode == 'test':
        print("Starting testing...")
        test_dataset = rnnDataset(paths['processed']['rgb']['encode']['test'],  
                         paths['processed']['pose']['encode']['test'], 
                         paths['processed']['combo']['csv']['test'], 
                         frame_select)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, 
                                    shuffle=False, num_workers=4)
    
        acc, loss = test(model, test_dataloader, args, device)
        print(f'Test Loss: {loss} | Test Accuracy: {acc}')
        

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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
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

    if save_filepath:
        # convert torch to CPU and then to NumPy
        encoding = torch.cat(all_scores).cpu().numpy()
        # save as NumPy file
        np.save(save_filepath, encoding)


    # Report accuracy and average loss
    acc = float(num_correct) / num_samples

    # Calculate average loss
    average_loss = np.mean(aggregate_loss)

    return acc, average_loss


if __name__ == "__main__":
    main()
