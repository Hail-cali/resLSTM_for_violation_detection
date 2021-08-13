import torch
import torch.nn as nn
import torchvision.models as models

resnet_50 = models.resnet50(pretrained=True)
resnet_50.fc = nn.Linear(resnet_50.fc.in_features, 200)

for param in resnet_50.parameters():
    param.requires_grad_(False)

