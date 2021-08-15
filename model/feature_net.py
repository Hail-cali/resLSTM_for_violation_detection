import torch
import torch.nn as nn
import torchvision.models as models

class FeatureNet(nn.Module):

    def __init__(self, batch=10, pretraind_model='resnet50'):
        super(FeatureNet, self).__init__()
        self.batch_size = batch
        # self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_pretrained_layer()
        #self.layer2 = self._make_layer()


    def _make_layer(self):
        layers = []
        #lstm
        return nn.Sequential(*layers)

    def _make_pretrained_layer(self):

        layer = models.resnet50(pretrained=True)
        layer.fc = nn.Linear(layer.fc.in_features, 200)

        for param in layer.parameters():
            param.requires_grad_(False)

        return layer

    def _forward_impl(self, x):
        #x = self.layer1(x)
        x = self.layer2(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


    def train_pretrained_layer(self, x):

        total_feature_map = []

        # d_size = len(x)
        # for f_size in range(0, d_size, self.batch_size):
        #     for frames in x[f_size:f_size + self.batch_size]:
        #         features = list()
        #         for frame in frames:
        #             input_data = torch.Tensor(frame.transpose(2, 0, 1)).unsqueeze(0)
        #             feature = layer.foward(input_data)
        #             features.append(feature)
        #
        #         total_feature_map.append(features)


        for frames in x:
            features = list()
            for frame in frames:
                input_data = torch.Tensor(frame.transpose(2, 0, 1)).unsqueeze(0)
                feature = self.layer1.forward(input_data)
                features.append(feature)

            total_feature_map.append(features)

        return total_feature_map


    def transfrom_video(self,x):

        return self.train_pretrained_layer(x)




