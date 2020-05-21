import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import collections

# Constants
C_NUM_CLASSES = 10

# Helper Functions
def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

# Helper Classes
# Flatten helper layer
class Flatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x, start_dim=1)

# LSTM model
class DefaultLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, num_classes)
        nn.init.kaiming_normal_(self.fc1.weight)
        
    def forward(self, x):
        # x has dimension (batch_size, seq_length, input_dim)
        # LSTM requires input of dimension (seq_length, batch_size, input_dim)
        x = torch.transpose(x, 0, 1)
        x, _ = self.lstm(x)
        # Only keep final LSTM output. Remaining dims in order (batch_size, hidden_dim)
        x = torch.squeeze(x[-1, :, :])
        # scores have dimension (batch_size, n_classes)
        scores = self.fc1(x)
        return scores

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
            ('fcfinal', nn.Linear(102, C_NUM_CLASSES))])
        )
        model.apply(init_weights)
        return model
    
    if model_name == "baseline_lstm":
        # number of features in input
        input_size = 102 + 512
        hidden_size = 100 # TO DO: we should make this a tunable paramter
        model = DefaultLSTM(input_size, hidden_size, C_NUM_CLASSES)
        return model
