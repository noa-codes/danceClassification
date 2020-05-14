import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def ModelChooser(model_name, **kwargs):
    """
    This function takes in a model name and returns its corresponding model
    Example usage:

        if model_name == "baseline_lstm":
            return LSTMLanguageModel(**kwargs)
    """
    if model_name == "resnet18":
        resnet18 = models.resnet18(pretrained=True, progress=True, **kwargs)
        return resnet18
