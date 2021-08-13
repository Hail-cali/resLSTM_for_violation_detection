import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import os
from utils import *

DPATH = '../data/fight'
FILE = 'fi001.mp4'

file_list = os.listdir(DPATH)
sample = os.path.join(DPATH, FILE)
print(file_list)

resnet_50 = models.resnet50(pretrained=True)
resnet_50.fc = nn.Linear(resnet_50.fc.in_features, 200)

for param in resnet_50.parameters():
    param.requires_grad_(False)

cap = cv2.VideoCapture(sample)
features = []
while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        input_data = torch.Tensor(frame.transpose(2, 0, 1)).unsqueeze(0)
        feature = resnet_50.forward(input_data)
        features.append(feature)
    else:
        print(f'{cap}: {frame}')
        break

cap.release()

total_features = torch.cat(features)

print(total_features.shape)
print(type(total_features))

total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)

print(total_frame)


total_frame = []

for name in file_list:
    path = os.path.join(DPATH, name)
    cap = cv2.VideoCapture(path)
    total_frame.append(cap.get(cv2.CAP_PROP_FRAME_COUNT))


print(total_frame)
