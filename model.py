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

    #Resnet18 model with identity final layer
    if model_name == "resnet18_feat":
        model = models.resnet18(pretrained=True)
        #Replace that final fc with identity layer
        model.fc = nn.Identity()
        return model