from torch.utils import data
from sklearn.model_selection import train_test_split



train = data.TensorDataset(train_x, train_y)
train_loader = data.DataLoader(train, batch_size = 64, shuffle = True)




val = data.TensorDataset(val_x, val_y)
val_loader = data.DataLoader(val, batch_size = 64, shuffle = True)