import pandas as pd
import numpy as np
from utils.data_loader import *
from models.feature_net import *
import torch.nn as nn
import torch.optim as optim
#import tensorboardX
from opts import parse_opts
from torch.utils import data
from opts import *

opt = parse_opts()
DPATH = '../dataset'
DEVICE = torch.device(f"cuda:{opt.gpu}" if opt.use_cuda else "cpu")
print(DEVICE, 'use')

# test mode loader
loader = DataLoader(path=DPATH, test_mode=True)

# train mode
# loader = DataLoader(path=DPATH)

X, y = loader.make_frame(mode='extract', device=DEVICE)

total_data = myDataset(x=X, y=y)

val_size = 0.5
train, val = data.random_split(total_data, [int(len(total_data)*val_size), len(total_data) - int(len(total_data)*val_size)])


train_loader = data.DataLoader(train, batch_size=2, shuffle=True)
val_loader = data.DataLoader(val, batch_size=2, shuffle=True)

model = ResLSTM()

#criterion = nn.BCEWithLogitsLoss()
criterion = nn.CrossEntropyLoss(reduction='sum')
optimizer = optim.Adam(model.parameters(), lr=0.001)



def train(model, optimizer, criterion, train_iter):

    model.train()
    for b, batch in enumerate(train_iter):
        x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        print(b, loss)

def evaluate(model, val_iter, criterion):
    model.eval()
    corrects, total_loss = 0, 0
    for batch in val_iter:
        x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
        y_pred = model(x)
        # logit = logit.type(torch.FloatTensor)
        # y = y.type(torch.FloatTensor)
        loss = criterion(y_pred, y)
        total_loss += loss.item()
        # corrects += (logit.max(1)[1].view(y.size()).data == y.data).sum

    size = len(val_iter.dataset)
    avg_loss = total_loss / size
    # avg_accuracy = 100.0*corrects / size
    return avg_loss  # , avg_accuracy


best_val_loss = None

for epoch in range(60):
    train(model, optimizer, criterion, train_loader)
    val_loss = evaluate(model, val_loader,criterion)

    print('[Epoch: %d] val loss : %5.2f ' % (epoch, val_loss))