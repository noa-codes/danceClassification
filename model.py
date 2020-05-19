import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import collections

# Flatten helper layer
class Flatten(nn.Module):
        def forward(self, x):
                    return flatten(x)

def ModelChooser(model_name, **kwargs):
    """
    This function takes in a model name and returns its corresponding model
    Example usage:

        if model_name == "baseline_lstm":
            return LSTMLanguageModel(**kwargs)
    """

    #Resnet18 model with identity final layer
    if model_name == "resnet18_features":
        model = models.resnet18(pretrained=True)
        #Replace that final fc with identity layer
        model.fc = nn.Identity()
        return model

    #Mini ConvNet to train on densepose features
    if model_name == "pose_features":
        model = nn.Sequential(collections.OrderedDict([
            ('conv1', nn.Conv2d(20, 64, 3, padding=1)), #input: 20x2x17, output: 64x2x17
            ('conv2', nn.Conv2d(64, 32, 3, padding=1)), #input: 64x2x17, output: 32x2x17
            ('flatten', Flatten()),
            ('fc1', nn.Linear(32*2*17, 512)),
            ('fc2', nn.Linear(512, 256)),
            ('fcfinal', nn.Linear(256, 10))])
        )
        return model
