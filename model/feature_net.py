import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
import numpy as np

class FeatureNet(nn.Module):

    def __init__(self, batch=10, pretraind_model='resnet50', class_num=2):
        super(FeatureNet, self).__init__()
        self.batch_size = batch
        # self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_pretrained_layer()
        self.layer2 = self._make_layer()
        self.fc1 = nn.Linear(80, 40)
        self.fc2 = nn.Linear(40, class_num)

    def _make_layer(self):
        layers = []
        # lstm
        lstm = nn.LSTM(input_size=200, hidden_size=80)

        layers.append(lstm)
        return nn.Sequential(*layers)

    def _make_pretrained_layer(self):

        layer = models.resnet50(pretrained=True)
        layer.fc = nn.Linear(layer.fc.in_features, 200)

        for param in layer.parameters():
            param.requires_grad_(False)

        # return layer
        return nn.Sequential(layer)

    def _forward_impl(self, f):
        #f ->  x_3d_list
        hidden = None
        x = self.forward_pretrained_layer(f)
        #print(x.shape)
        print(type(x[0]))
        out, hidden = self.layer2(x)

        x = self.fc1(out[-1, :, :])
        #x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x)
        return x


    def forward(self, x):
        return self._forward_impl(x)

    def forward_pretrained_layer(self, frames):
        feature_map = []
        for frame in frames:
            input_data = torch.Tensor(frame.transpose(2, 0, 1)).unsqueeze(0)
            feature = self.layer1.forward(input_data)
            feature_map.append(feature)

        #return np.vstack(feature_map)
        #return feature_map

        print(f'feature map att type {type(feature_map[0])}')
        # print(feature_map[0].shape)
        # temp = torch.stack(feature_map)
        # torch.Tensor(feature_map).transpose(2, 0, 1).unsqueeze(0)
        #return pack_padded_sequence(temp, [2, 0, 1])
        return torch.stack(feature_map)

    def train_pretrained_layer(self, x):

        total_feature_map = []

        for frames in x:
            features = list()
            for frame in frames:
                input_data = torch.Tensor(frame.transpose(2, 0, 1)).unsqueeze(0)
                feature = self.layer1.forward(input_data)
                features.append(feature)

            total_feature_map.append(features)

        return total_feature_map

    def transfrom_video(self, x):
        return self.train_pretrained_layer(x)

class ResLSTM(FeatureNet):

    def set_config(self):
        pass


if __name__=='__main__':
    f = FeatureNet()
