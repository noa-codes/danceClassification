import torch
import torch.nn as nn
import torch.nn.functional as F


def ModelChooser(model_name, **kwargs):
    """
    This function takes in a model name and returns its corresponding model
    Example usage:
    
        if model_name == "baseline_lstm":
            return LSTMLanguageModel(**kwargs)
    """
    pass 