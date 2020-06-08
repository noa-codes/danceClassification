import os
import time
import torch
import matplotlib.pyplot as plt
from dataset import *

def create_unique_logdir(logdir, lr, root_logdir="/mnt/disks/disk1/log/"):
    """
    Creates a unique log directory using the directory name and the time stamp
    Takes in a unqiue directory name and optionally a root directory path
    The root directory path is default to "log/" since all logs should be stored
    under that directory

    Example:
        > create_unique_logdir("baseline_lstm")
        "log/baseline_lstm_2020_2_27_h16_m5_lr3e-4"
    """
    if logdir == "":
        return logdir
    localtime = time.localtime(time.time())
    time_label = "{}_{}_{}_h{}_m{}_lr{}".format(localtime.tm_year, localtime.tm_mon, \
        localtime.tm_mday, localtime.tm_hour, localtime.tm_min, lr)
    unique_logdir = os.path.join(root_logdir, logdir + "_" + time_label)
    os.makedirs(unique_logdir, exist_ok=True)
    return unique_logdir


def save_checkpoint(logdir, model, optimizer, epoch, loss, lr, best=None):
    """
    Saves model checkpoint after each epoch

    best: An optional string used to specify which validation method this best
    checkpoint is for
    """
    checkpoint_dir = os.path.join(logdir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    if best:
        checkpoint_path = "{}/best_{}.pth".format(checkpoint_dir, best)
    else:
        checkpoint_path = "{}/lr{}_epoch{}.pth".format(checkpoint_dir, lr, epoch)

    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, checkpoint_path)

    if not best:
        print("Saving checkpoint to lr{}_epoch{}".format(lr, epoch))


def load_checkpoint(model_checkpoint, model, device, optimizer=None):
    """
    Loads a pretrained checkpoint to continue training
    model_checkpoint: Path of the model_checkpoint that ends with .pth
    """
    checkpoint = torch.load(model_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.to(device)
    return model


def make_paths(raw_data_path, proc_data_path):
    # Dictionary to paths nesting as follows:
    paths = {'raw': {'rgb': '', 'pose': ''},
             'processed': {
                'rgb': {
                    'encode': {'train': '', 'val': '', 'test': ''}
                },
                'pose': {
                    'encode': {'train': '', 'val': '', 'test': ''}
                },
                'combo' : {
                    'encode': '',
                    'csv': {'train': '', 'val': '', 'test': ''}
                }
             }
            }

    paths['raw']['rgb'] = os.path.join(raw_data_path, 'rgb')
    paths['raw']['pose'] = os.path.join(raw_data_path, 'densepose')
    paths['processed']['rgb']['encode']['train'] = os.path.join(proc_data_path,
                                                            "rgb/encoded_features_train.npy")
    paths['processed']['rgb']['encode']['val'] = os.path.join(proc_data_path,
                                                                  "rgb/encoded_features_val.npy")
    paths['processed']['rgb']['encode']['test'] = os.path.join(proc_data_path, "rgb/encoded_features_test.npy")

    paths['processed']['pose']['encode']['train'] = os.path.join(proc_data_path, "densepose/encoded_features_train.npy")
    paths['processed']['pose']['encode']['val'] = os.path.join(proc_data_path, "densepose/encoded_features_val.npy")
    paths['processed']['pose']['encode']['test'] = os.path.join(proc_data_path, "densepose/encoded_features_test.npy")
    paths['processed']['combo']['encode'] = os.path.join(proc_data_path, "combo")
    paths['processed']['combo']['csv']['train'] = os.path.join(proc_data_path, C_TRAIN_CSV)
    paths['processed']['combo']['csv']['val'] = os.path.join(proc_data_path, C_VAL_CSV)
    paths['processed']['combo']['csv']['test'] = os.path.join(proc_data_path, C_TEST_CSV)
    return paths