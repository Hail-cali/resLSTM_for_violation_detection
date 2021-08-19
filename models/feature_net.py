import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
import numpy as np

class FeatureNet(nn.Module):

    def __init__(self, batch=10, pretraind_model='resnet50', class_num=2):
        super(FeatureNet, self).__init__()
        self.class_num = class_num
        self.batch_size = batch

        self.layer1 = self._make_pretrained_layer()
        self.layer2 = self._make_layer()
        self.fc1 = nn.Linear(80, 40)
        self.fc2 = nn.Linear(40, self.class_num)


    def _make_layer(self):
        layers = []
        lstm = nn.LSTM(input_size=200, hidden_size=80)
        layers.append(lstm)
        return nn.Sequential(*layers)

    def _make_pretrained_layer(self):

        layer = models.resnet50(pretrained=True)
        layer.fc = nn.Linear(layer.fc.in_features, 200)

        for param in layer.parameters():
            param.requires_grad_(False)

        return nn.Sequential(layer)

    def _forward_impl(self, f):

        #f ->  x_3d_list
        hidden = None
        x = self.forward_pretrained_layer(f)
        out, hidden = self.layer2(x)
        x = self.fc1(out[-1, :, :])
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x

    def forward(self, x):
        return self._forward_impl(x)

    def forward_pretrained_layer(self, frames):
        feature_map = []
        for frame in frames:
            input_data = torch.Tensor(frame.transpose(2, 0, 1)).unsqueeze(0)
            feature = self.layer1.forward(input_data)
            feature_map.append(feature)

        return torch.stack(feature_map)

class ResLSTM(nn.Module):

    def __init__(self):
        super(ResLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=200,
                            hidden_size=80,
                            bidirectional=True,
                            batch_first=True)
        self.fc = nn.Linear(160, 1)

    def forward(self, X):
        outputs, _ = self.lstm(X)
        outputs = outputs[:, -1, :]
        return self.fc(outputs)

class lLSTM(nn.Module):
    """
    #
    """
    def __init__(self):
        super(lLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=200,
                            hidden_size=80,
                            bidirectional=True,
                            batch_first=True)
        self.fc = nn.Linear(160, 1)

    def forward(self, X):
        outputs, _ = self.lstm(X)
        outputs = outputs[:, -1, :]
        return self.fc(outputs)

