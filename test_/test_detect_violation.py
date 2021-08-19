import pandas as pd
import numpy as np
from utils.data_loader import *
from models.feature_net import *
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
import tensorboardX
from opts import parse_opts
from torch.utils import data

DPATH = '../dataset'

# test mode loader
loader = DataLoader(path=DPATH, test_mode=True)

# train mode
# loader = DataLoader(path=DPATH)

X, y = loader.make_frame(mode='extract')

total_data = myDataset(x=X, y=y)

val_size = 0.5
train, val = data.random_split(total_data, [int(len(total_data)*val_size), len(total_data) - int(len(total_data)*val_size)])


train_loader = data.DataLoader(train, batch_size=1, shuffle=True)
val_loader = data.DataLoader(val, batch_size=1, shuffle=True)

model = lLSTM()

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

def train(model, optimizer, train_iter):
    model.train()
    for b, batch in enumerate(train_iter):
        x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)

        optimizer.zero_grad()
        logit = model(x)
        # logit = logit.type(torch.FloatTensor)
        # y = y.type(torch.FloatTensor)
        loss = criterion(logit, y)
        loss.backward()
        optimizer.step()

def evaluate(model, val_iter):
    model.eval()
    corrects, total_loss = 0, 0
    for batch in val_iter:
        x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
        logit = model(x)
        # logit = logit.type(torch.FloatTensor)
        # y = y.type(torch.FloatTensor)
        loss = criterion(logit, y)
        total_loss += loss.item()
        # corrects += (logit.max(1)[1].view(y.size()).data == y.data).sum

    size = len(val_iter.dataset)
    avg_loss = total_loss / size
    # avg_accuracy = 100.0*corrects / size
    return avg_loss  # , avg_accuracy


best_val_loss = None

for epoch in range(60):
    train(model, optimizer, train_loader)
    val_loss = evaluate(model, val_loader)

    print('[Epoch: %d] val loss : %5.2f ' % (epoch, val_loss))