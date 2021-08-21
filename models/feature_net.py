import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np

#torch.no_grad()

class FeatureNet(nn.Module):

    def __init__(self, batch=10, pretraind_model='resnet50', class_num=2):
        super(FeatureNet, self).__init__()
        self.class_num = class_num
        self.batch_size = batch
        self.layer1 = self._make_pretrained_layer()
        self.layer2 = self._make_layer()
        self.fc1 = nn.Linear(256, 80)
        self.fc2 = nn.Linear(80, self.class_num)

    def _make_layer(self):
        layers = []
        lstm = nn.LSTM(input_size=300, hidden_size=256)
        layers.append(lstm)
        return nn.Sequential(*layers)

    def _make_pretrained_layer(self):

        layer = models.resnet50(pretrained=True)
        layer.fc = nn.Linear(layer.fc.in_features, 300)
        # for param in layer.parameters():
        #     param.requires_grad_(False)
        return nn.Sequential(layer)

    def _forward_impl(self, f):
        hidden = None
        x = self.forward_pretrained_layer(f)
        out, hidden = self.layer2(x.to('cuda'))
        x = self.fc1(out[:, -1, :])   # shape: torch.Size([1, 80, 200])
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x

    def forward(self, x):
        return self._forward_impl(x)

    def forward_pretrained_layer(self, batch_frames):
        batch_f = []
        for frames in batch_frames:
            feature_map = []
            for frame in frames:
                # print(type(frame))<class 'numpy.ndarray'>
                # print(frame.shape) (360, 640, 3)
                frame = np.array(frame.cpu())
                input_data = torch.Tensor(frame.transpose(2, 0, 1)).unsqueeze(0).to('cuda')
                # input data shpae torch.Size([1, 3, 360, 640])
                feature = self.layer1.forward(input_data)
                feature = feature.cpu().detach().numpy()

                feature_map.append(feature)
            batch_f.append(feature_map)

        return torch.Tensor(batch_f).squeeze(2)

class ResLSTM(nn.Module):

    def __init__(self, class_num=2):
        super(ResLSTM, self).__init__()
        self.class_num = class_num
        self.layer2 = self._make_layer()
        self.fc1 = nn.Linear(80, 40)
        self.bn = nn.BatchNorm1d(40)
        self.fc2 = nn.Linear(40, self.class_num)

    def _make_layer(self):
        layers = []
        lstm = nn.LSTM(input_size=200, hidden_size=80,
                       batch_first=True)
        layers.append(lstm)
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        hidden = None
        out, hidden = self.layer2(x)
        x = self.fc1(out[:, -1, :])
        x = F.relu(x)
        #x = self.bn(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x

    def forward(self, X):
        return self._forward_impl(X)

