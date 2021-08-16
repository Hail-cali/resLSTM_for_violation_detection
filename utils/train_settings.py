

def train_epoch(model, data_loader, criterion, optimizer , epoch, log_interval, device):
    model.train()

    for batch_idx, (X,y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)

        y_pred = model.forward()
        loss = criterion(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()