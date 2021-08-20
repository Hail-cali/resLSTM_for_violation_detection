# jinyong's model arc

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.utils.data as data
import numpy as np
import glob
import cv2
import os

class myDataset(data.Dataset):
    def __init__(self, x, y, x_len):
        super(myDataset, self).__init__()

        self.x = x
        self.y = y
        self.x_len = x_len

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.x_len[index]

    def __len__(self):
        return len(self.x)


class Resnet_feature_extract(nn.Module):
    def __init__(self, emb_size=300, dropout=0.3):
        super(Resnet_feature_extract, self).__init__()

        self.dropout = dropout

        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(resnet.fc.in_features, 512)
        self.bn1 = nn.BatchNorm1d(512, momentum=0.01)
        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512, momentum=0.01)
        self.fc3 = nn.Linear(512, emb_size)

    def forward(self, x):
        emb_seq = []
        # X = x
        for t in range(x.size(1)):
            with torch.no_grad():
                x = self.resnet(x[:, t, :, :, :])
                x = x.view(x.size(0), -1)

            x = self.bn1(self.fc1(x))
            x = F.relu(x)
            x = self.bn2(self.fc2(x))
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.fc3(x)

            emb_seq.append(x)

        emb_seq = torch.stack(emb_seq, dim=0).transpose_(0, 1)

        return emb_seq


class LSTM(nn.Module):
    def __init__(self, feature_dim=300, num_layers=3, hidden_size=256, fc_dim=128, dropout=0.3, num_classes=2):
        super(LSTM, self).__init__()

        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.fc_dim = fc_dim
        self.dropout = dropout
        self.num_classes = num_classes

        self.lstm = nn.LSTM(input_size=self.feature_dim,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True)
        self.fc1 = nn.Linear(self.hidden_size, self.fc_dim)
        self.fc2 = nn.Linear(self.fc_dim, self.num_classes)

    def forward(self, x_RNN, x_lengths):
        N, T, n = x_RNN.size()

        for i in range(N):
            if x_lengths[i] < T:
                x_RNN[i, x_lengths[i]:, :] = torch.zeros(T - x_lengths[i], n, dtype=torch.float, device=x_RNN.device)

        lengths_ordered, perm_idx = x_lengths.sort(0, descending=True)

        packed_x_RNN = torch.nn.utils.rnn.pack_padded_sequence(x_RNN[perm_idx], lengths_ordered, batch_first=True)
        self.lstm.flatten_parameters()
        packed_rnn_out, _ = self.lstm(packed_x_RNN, None)

        rnn_out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_rnn_out, batch_first=True)
        rnn_out = rnn_out.contiguous()

        _, unperm_idx = perm_idx.sort(0)
        rnn_out = rnn_out[unperm_idx]

        x = self.fc1(rnn_out[:, -1, :])
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)

        return x


def generate_data(folder_list, max_frame=100):
    total_video = []
    video_len = []
    for path in folder_list:
        directory_path = "./" + path + "/*"
        video_list = glob.glob(directory_path)

        for video_path in video_list:
            cap = cv2.VideoCapture(video_path)

            num_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            frame_list = np.zeros((max_frame, 3, 224, 224), dtype=np.float)

            video_len.append(min(max_frame, num_frame))

            while (cap.isOpened()):
                ret, frame = cap.read()
                current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

                if current_frame == max_frame:
                    break

                if ret:
                    frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
                    frame = frame.transpose(2, 0, 1)
                    frame_list[current_frame] = frame
                else:
                    break

            cap.release()
            if len(frame_list) > max_frame:
                frame_list = frame_list[max_frame]

            total_video.append(frame_list)

        if path == 'fight':
            fight_size = len(video_list)
        elif path == 'noFight':
            nonfight_size = len(video_list)

    label = np.vstack((np.repeat(np.array(1.0), fight_size, axis=0), np.repeat(np.array(0.), nonfight_size, axis=0)))
    total_video = np.array(total_video)
    return torch.FloatTensor(total_video), torch.FloatTensor(label.reshape(-1, 1)), torch.Tensor(
        np.array(video_len).reshape(-1, 1))


def train(model, train_loader, optimizer, epoch, device):
    resnet_model, lstm_model = model
    resnet_model.train()
    lstm_model.train()

    epoch_loss, all_y, all_y_pred = 0, [], []
    N_count = 0
    for batch_idx, (x, y, x_len) in enumerate(train_loader):
        x, y, x_len = x.to(device), y.to(device).view(-1, ), x_len.to(device).view(-1, )
        N_count += x.size(0)

        optimizer.zero_grad()
        res_output = resnet_model(x)
        output = lstm_model(res_output, x_len)

        loss = F.cross_entropy(output, y)
        epoch_loss += F.cross_entropy(output, y, reduction='sum').item()

        y_pred = torch.max(output, 1)[1]

        all_y.extend(y)
        all_y_pred.extend(y_pred)

        step_score = (y == y_pred).to(torch.float).mean()

        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 5 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.2f}%'.format(
                epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item(),
                100 * step_score
            ))

    epoch_loss /= len(train_loader)

    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    epoch_score = (all_y == all_y_pred).to(torch.float).mean()

    return epoch_loss, epoch_score


def validation(model, device, optimizer, test_loader):
    save_model_path = "./weights"

    resnet_model, lstm_model = model
    resnet_model.eval()
    lstm_model.eval()

    test_loss = 0
    all_y, all_y_pred = [], []
    with torch.no_grad():
        for x, y, x_len in test_loader:
            x, y, x_len = x.to(device), y.to(device).view(-1, ), x_len.to(device).view(-1, )

            output = lstm_model(resnet_model(x), x_len)

            loss = F.cross_entropy(output, y, reduction='sum')
            test_loss += loss.item()
            y_pred = output.max(1, keepdim=True)[1]

            all_y.extend(y)
            all_y_pred.extend(y_pred)

    test_loss /= len(test_loader.dataset)

    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    test_score = (all_y == all_y_pred).to(torch.float).mean()

    print('\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(len(all_y), test_loss,
                                                                                        100 * test_score))

    check_mkdir(save_model_path)
    torch.save(lstm_model.state_dict(),
               os.path.join(save_model_path, 'cnn_encoder_epoch{}.pth'.format(epoch + 1)))  # save spatial_encoder
    torch.save(resnet_model.state_dict(),
               os.path.join(save_model_path, 'rnn_decoder_epoch{}.pth'.format(epoch + 1)))  # save motion_encoder
    torch.save(optimizer.state_dict(),
               os.path.join(save_model_path, 'optimizer_epoch{}.pth'.format(epoch + 1)))  # save optimizer
    print("Epoch {} model saved!".format(epoch + 1))

    return test_loss, test_score


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


if __name__ == "__main__":
    # path_list = ["fight", "noFight"]

    # raw_x, y, video_len = generate_data(path_list)

    # total_data = myDataset(x = raw_x, y = y, x_len = video_len)
    # train_data, val_data = data.random_split(total_data,[int(len(total_data)*0.8), len(total_data) - int(len(total_data)*0.8)])

    train_data = torch.load('./processed/train_data.pkl')
    val_data = torch.load('./processed/val_data.pkl')

    train_loader = data.DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = data.DataLoader(val_data, batch_size=32, shuffle=True)

    use_cuda = torch.cuda.is_available()  # check if GPU exists
    device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU

    resnet_model = Resnet_feature_extract().to(device)
    lstm_model = LSTM().to(device)

    if torch.cuda.device_count() > 1:
        resnet_model = nn.DataParallel(resnet_model)
        lstm_model = nn.DataParallel(lstm_model)
        crnn_params = list(resnet_model.module.fc1.parameters()) + list(resnet_model.module.bn1.parameters()) + \
                      list(resnet_model.module.fc2.parameters()) + list(resnet_model.module.bn2.parameters()) + \
                      list(resnet_model.module.fc3.parameters()) + list(lstm_model.parameters())

    elif torch.cuda.device_count() == 1:
        crnn_params = list(resnet_model.fc1.parameters()) + list(resnet_model.bn1.parameters()) + \
                      list(resnet_model.fc2.parameters()) + list(resnet_model.bn2.parameters()) + \
                      list(resnet_model.fc3.parameters()) + list(lstm_model.parameters())

    optimizer = torch.optim.Adam(crnn_params, lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=15, min_lr=1e-10, verbose=True)

    epoch_train_losses = []
    epoch_train_scores = []
    epoch_test_losses = []
    epoch_test_scores = []

    epochs = 50
    for epoch in range(epochs):
        epoch_train_loss, epoch_train_score = train([resnet_model, lstm_model], train_loader, optimizer, epoch, device)
        epoch_test_loss, epoch_test_score = validation([resnet_model, lstm_model], device, optimizer, val_loader)
        scheduler.step(epoch_test_loss)

        epoch_train_losses.append(epoch_train_loss)
        epoch_train_scores.append(epoch_train_score)
        epoch_test_losses.append(epoch_test_loss)
        epoch_test_scores.append(epoch_test_score)

        epoch_train_losses = np.array(epoch_train_losses)
        epoch_train_scores = np.array(epoch_train_scores)
        epoch_test_losses = np.array(epoch_test_losses)
        epoch_test_scores = np.array(epoch_test_scores)

        np.save('./results/epoch_train_losses.npy', epoch_train_losses)
        np.save('./results/epoch_train_scores.npy', epoch_train_scores)
        np.save('./results/epoch_test_losses.npy', epoch_test_losses)
        np.save('./results/epoch_test_scores.npy', epoch_test_scores)