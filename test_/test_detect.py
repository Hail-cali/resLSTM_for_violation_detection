import pandas as pd
import numpy as np
from utils.data_loader import *
from model.feature_net import *
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
DPATH = '../data/fight'


loader = DataLoader(path=DPATH)
print(loader.file_list)
total_frame = loader.make_frame(mode='train')

print(f'total_frame len: {len(total_frame)}')
print([len(frames) for frames in total_frame])

print([frames[-1].shape for frames in total_frame])

model = FeatureNet()

sample = total_frame[10]

#x = model.forward(sample)

#print(x)

criterion = nn.CrossEntropyLoss(reduction='sum')
#optimizer = optim.SGD(model.parameters(), lr=0.1)
optimizer = optim.Adam(model.parameters(), lr=0.001)
epoch = 10
y = torch.LongTensor([1])
#print(f'result : {summary(model)}')
for t in range(epoch):

    y_pred = model.forward(sample)
    loss = criterion(y_pred, y)

    if t % 2000 == 1999:
        print(f'step {t} | loss : {loss.item()}')

    print(f'step {t} || loss : {loss.item()}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f'result : {summary(model)}')


# print(model.summary)
print()