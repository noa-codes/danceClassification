import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import collections

# Flatten helper layer
class Flatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x, start_dim=1)

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

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
    # Taken from https://arxiv.org/pdf/1909.03466.pdf
    # Multi-Modal Three-Stream Network for Action Recognition
    # conv-conv-relu-maxpool-fc-softmax
    if model_name == "pose_features":
        model = nn.Sequential(collections.OrderedDict([
            ('conv1', nn.Conv2d(20, 3, 3, padding=1)), #input: 20x2x17, output: 3x2x17
            ('conv2', nn.Conv2d(3, 3, 3, padding=1)), #input: 3x2x17, output: 3x2x17
            ('flatten', Flatten()),
            ('relu', nn.ReLU()),
#             ('pool', nn.MaxPool2d(2, 2)), #input: 3x2x17, 
            ('fcfinal', nn.Linear(102, 10))])
        )
        model.apply(init_weights)
        return model
