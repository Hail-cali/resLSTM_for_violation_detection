from torch.utils import data

train = data.TensorDataset(train_x, train_y)
train_loader = data.DataLoader(train, batch_size = 64, shuffle = True)

val = data.TensorDataset(val_x, val_y)
val_loader = data.DataLoader(val, batch_size = 64, shuffle = True)