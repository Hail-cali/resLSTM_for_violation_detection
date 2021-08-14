import torch
import torch.nn as nn
import torchvision.models as models

class FeatureNet(nn.Module):

    def __init__(self, D_in, H, D_out ,batch=10, pretraind_model='resnet50'):
        self.batch_size =batch
        self.
        # self.X = None
        # self.W = None
        if pretraind_model == 'resnet50':
            self.models = models.resnet50(pretrained=True)
        else:
            self.models = None




    def forward(self, x):

        pass



resnet_50 = models.resnet50(pretrained=True)
resnet_50.fc = nn.Linear(resnet_50.fc.in_features, 200)

for param in resnet_50.parameters():
    param.requires_grad_(False)



# for frames in total_frame:
#     features = []
#     for frame in frames:
#         input_data = torch.Tensor(frame.transpose(2, 0, 1)).unsqueeze(0)
#         feature = resnet_50.forward(input_data)
#         features.append(feature)

