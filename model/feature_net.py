import torch
import torch.nn as nn
import torchvision.models as models



class FeatureNet(nn.Module):

    def __init__(self, batch=10, pretraind_model='resnet50', class_num=2):
        super(FeatureNet, self).__init__()
        self.batch_size = batch
        # self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_pretrained_layer()
        self.layer2 = self._make_layer()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, class_num)


    def _make_layer(self):
        layers = []
        # lstm
        lstm = nn.LSTM(input_size=200, hidden_size=256, num_layers=3)

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
        x = self.forward_pretrained_layer(f)
        #x = self.layer2(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

    def forward_pretrained_layer(self, frames):
        feature_map = []
        for frame in frames:
            input_data = torch.Tensor(frame.transpose(2, 0, 1)).unsqueeze(0)
            feature = self.layer1.forward(input_data)
            feature_map.append(feature)

        return feature_map

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
