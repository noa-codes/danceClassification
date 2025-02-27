import collections
from model.tcn import TemporalConvNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math

# Constants
C_NUM_CLASSES = 10
C_INPUT_SIZE = 102 + 512

# Helper Classes
# Flatten helper layer
class Flatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x, start_dim=1)


class DefaultLSTM(nn.Module):
    """ Baseline LSTM model with one LSTM layer and one linear layer
    """
    def __init__(self, input_size, hidden_size, num_classes, dropout_rate=0, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout_rate)
        self.fc1 = nn.Linear(hidden_size, num_classes)
        self.drop = nn.Dropout(p=dropout_rate)
        nn.init.kaiming_normal_(self.fc1.weight)

    def forward(self, x):
        # x has dimension (batch_size, seq_length, input_dim)
        # LSTM requires input of dimension (seq_length, batch_size, input_dim)
        x = torch.transpose(x, 0, 1)
        x = self.drop(x)
        x, _ = self.lstm(x)
        # Only keep final LSTM output. Remaining dims in order (batch_size, hidden_dim)
        x = torch.squeeze(x[-1, :, :])
        # scores have dimension (batch_size, n_classes)
        scores = self.fc1(x)
        return scores

class MultiLayerLSTM(nn.Module):
    """ Multilayer LSTM model with configuralbe number LSTM layer and one linear layer
    """
    def __init__(self, input_size, hidden_size, num_classes, num_layers, dropout_rate=0):
        super().__init__()
        self.lstms = nn.ModuleList([nn.LSTM(input_size, hidden_size)])
        self.lstms.extend([nn.LSTM(hidden_size, hidden_size) for i in range(num_layers-1)])
        self.fc1 = nn.Linear(hidden_size, num_classes)
        self.drop = nn.Dropout(p=dropout_rate)
        nn.init.kaiming_normal_(self.fc1.weight)

    def forward(self, x):
        # x has dimension (batch_size, seq_length, input_dim)
        # LSTM requires input of dimension (seq_length, batch_size, input_dim)
        x = torch.transpose(x, 0, 1)
        x = self.drop(x)
        for layer in self.lstms:
            x, _ = layer(x)
        # Only keep final LSTM output. Remaining dims in order (batch_size, hidden_dim)
        x = torch.squeeze(x[-1, :, :])
        # scores have dimension (batch_size, n_classes)
        scores = self.fc1(x)
        return scores


class AttentionLSTM(nn.Module):
    """ LSTM model with attention
    """
    def __init__(self, input_size, hidden_size, num_classes, dropout_rate=0):
        super().__init__()
        self.scale = 1. / math.sqrt(hidden_size)
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.decoder = nn.Linear(hidden_size, num_classes)
        nn.init.kaiming_normal_(self.decoder.weight)

    def forward(self, x, hidden=None):
        ### 1) Encoder
        # x has dimension (batch_size, seq_length, input_dim)
        # LSTM requires input of dimension (seq_length, batch_size, input_dim)
        x = torch.transpose(x, 0, 1)
        x = self.dropout(x)
        outputs, hidden = self.lstm(x)
        # `hidden` contains both the hidden state & cell state
        # extract just the cell state
        hidden = hidden[1]
        # get the final cell state
        hidden = hidden[-1]

        ### 2) Attention
        # (batch, hidden_size) -> (batch, 1, hidden_size)
        query = hidden.unsqueeze(1)
        # (seq_length, batch, hidden_size) -> (batch, hidden_size, seq_length)
        keys = outputs.transpose(0,1).transpose(1,2)
        # (batch, 1, hidden_size) x (batch, hidden_size, seq_length) -> (batch, 1, seq_length)
        energy = torch.bmm(query, keys)
        # scale and normalize
        energy = F.softmax(energy.mul_(self.scale), dim=2)

        # (seq_length, batch, hidden_size) -> (batch, seq_length, hidden_size)
        values = outputs.transpose(0,1)
        # (batch, 1, seq_length) x (batch, seq_length, hidden_size) -> (batch, hidden_size)
        linear_combination = torch.bmm(energy, values).squeeze(1)

        ### 3) Classifier
        # logits (i.e., scores) have dimension (batch_size, num_classes)
        logits = self.decoder(linear_combination)
        return logits

class PoseCNN(nn.Module):
    """
    Taken from https://arxiv.org/pdf/1909.03466.pdf,
    Multi-Modal Three-Stream Network for Action Recognition
    Structure: conv-conv-relu-maxpool-fc-softmax
    """
    def __init__(self):
        super(PoseCNN, self).__init__()
        self.network = nn.Sequential(collections.OrderedDict([
            ('conv1', nn.Conv2d(20, 3, 3, padding=1)), #input: 20x2x17, output: 3x2x17
            ('conv2', nn.Conv2d(3, 3, 3, padding=1)), #input: 3x2x17, output: 3x2x17
            ('flatten', Flatten()),
            ('relu', nn.ReLU()),
            ('fcfinal', nn.Linear(102, C_NUM_CLASSES))])
        )
        model.apply(init_weights)

    def init_weights(self, m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    """
    Temporal Convolutional Network, adapted from Bai et al (2018)
    https://github.com/locuslab/TCN/blob/master/TCN/mnist_pixel/model.py
    """
    def __init__(self, input_size, num_classes, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels,
            kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x):
        # x has dimension (batch_size, seq_length, input_dim)
        ## TCN requires input of dimenson (batch_size, input_dim, seq_length)
        x = torch.transpose(x, 1, 2)
        x = self.tcn(x)
        # only keep final TCN output for linear layer
        scores = self.linear(x[:, :, -1])
        return scores


def ModelChooser(model_name, args):
    """
    This function takes in a model name and returns its corresponding model
    Example usage:

        if model_name == "baseline_lstm":
            return LSTMLanguageModel(**kwargs)
    """

    # Resnet18 model with identity final layer
    if model_name == "resnet18_features":
        model = models.resnet18(pretrained=True)
        # Replace that final fc with identity layer
        model.fc = nn.Identity()
        is_frame_by_frame = True

    # Mini ConvNet to train on densepose features
    if model_name == "pose_features":
        model = PoseCNN()
        is_frame_by_frame = True

    # Simple LSTM model
    if model_name == "baseline_lstm":
        model = DefaultLSTM(C_INPUT_SIZE, args.hidden_size,
                            C_NUM_CLASSES, dropout_rate=args.dropout, num_layers=args.num_layers)
        is_frame_by_frame = False

    # LSTM model with attention
    if model_name == "attention_lstm":
        model = AttentionLSTM(C_INPUT_SIZE, args.hidden_size,
                              C_NUM_CLASSES, dropout_rate=args.dropout)
        is_frame_by_frame = False

    # Baseline CNN frame by frame
    if model_name == "baseline_cnn":
        model = nn.Sequential(collections.OrderedDict([
            ('fc1', nn.Linear(C_INPUT_SIZE, args.hidden_size)),
            ('dropout', nn.Dropout(p=args.dropout)),
            ('relu', nn.ReLU()),
            ('fcfinal', nn.Linear(args.hidden_size, C_NUM_CLASSES)), ])
        )
        # initialize weights
        nn.init.kaiming_normal_(model.fc1.weight)
        nn.init.kaiming_normal_(model.fcfinal.weight)
        is_frame_by_frame = True

    # Temporal Convolutional Network
    if model_name == 'tcn':
        channel_sizes = [args.hidden_size] * args.levels
        model = TCN(
            input_size=C_INPUT_SIZE,
            num_classes=C_NUM_CLASSES,
            num_channels=channel_sizes,
            kernel_size=3,
            dropout=args.dropout)
        is_frame_by_frame = False

    return model, is_frame_by_frame
