import pandas as pd
import numpy as np
import os
import cv2
from collections import OrderedDict
from itertools import chain
import torch
import torch.utils.data as data

class myDataset(data.Dataset):
    """
    #
    """
    def __init__(self, x, y):
        super(myDataset, self).__init__()
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


class DataLoader(object):

    def __init__(self, path=None, labels=[1, 0], test_mode=False, batch_size=64, mode='train',
                 img_resize=False, test_size=0.3, shuffle=True,
                 verbose=False):
        self.path = path
        self.labels = labels
        self.batch_size = batch_size
        self.mode = mode
        self.verbose = verbose
        self.img_resize = img_resize

        self.test_size = test_size
        self.shuffle = shuffle
        self.test_mode = test_mode

        if not self.test_mode:
            try:
                self.file_dir = [os.path.join(self.path, label) for label in os.listdir(self.path)]

                self.file_list = [[os.path.join(f,file) for file in os.listdir(f)] for f in self.file_dir]
                print(f'file size: {tuple(len(file) for file in self.file_list)} ex) {self.file_list[0][0]}')
            except FileNotFoundError:
                print(f'[Errno 2] No such file or directory: {self.path}')
                self.file_list = None
        else:
            # test_mode == True
            try:
                self.file_dir = [os.path.join(self.path, label) for label in os.listdir(self.path)]

                self.file_list = [[os.path.join(f, file) for file in os.listdir(f)][:1] for f in self.file_dir]
                print(f'file size: {tuple(len(file) for file in self.file_list)} ex) {self.file_list[0][0]}')
            except FileNotFoundError:
                print(f'[Errno 2] No such file or directory: {self.path}')
                self.file_list = None

        self.data_size = len(self.file_list)

    def split_train_test_data(self, X, y, test_size=0.1):
        from sklearn.model_selection import train_test_split
        return train_test_split(X, y, test_size=test_size)


    def _video_to_frame(self, file_name):
        """

        :param file_name: file path + file name
        :return: frame for single video :type: [torch.Tensor] :shape: [80, 360, 640, 3]
        """
        filepath = os.path.join(self.path, file_name)
        cap = cv2.VideoCapture(filepath)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                if self.img_resize:
                    frame = cv2.resize(frame)
                frames.append(frame)
            else:
                #print(f'{cap}: {frame}')
                break
        cap.release()
        frames = torch.Tensor(frames).squeeze(1)

        # return frames
        if frames.shpae[0] >= 80:
            return frames[:80]

        elif frames.shpae[0] < 80:
            zero_padding = torch.Tensor([np.zeros(frames[0].shape)]*(80-frames.shape[0]))
            return torch.vstack([frames,zero_padding])



    def __video_to_frame(self, file_name, model=None):
        """
        :param file_name: filepath + filename
        :param model: pretrained model class which fine tuned
        :return: feature map for single video :type: [list, torch.Tensor(1d array)]
        """
        filepath = os.path.join(self.path, file_name)
        cap = cv2.VideoCapture(filepath)

        frames = []
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                input_data = torch.Tensor(frame.transpose(2, 0, 1)).unsqueeze(0)
                feature = model.forward(input_data)
                frames.append(feature.detach().numpy())
            else:
                break

        cap.release()

        # frames = torch.stack(frames)
        frames = torch.Tensor(frames).squeeze(1)
        if len(frames) < 80:
            padding = torch.Tensor(np.repeat(np.expand_dims(np.zeros((200)), 0), 80 - len(frames), axis=0))
            frames = torch.vstack((frames, padding))
        elif len(frames) >= 80:
            frames = frames[:80]

        if self.verbose:
            print(f'len frame: {len(frames)} shape {frames[0].shape}')
        print(f'len frame: {len(frames)} shape {frames[0].shape}')
        return frames


    def make_frame(self, mode='train', output_shape=(360, 640, 3), verbose=False):
        """
        :param verbose:
        :param mode: [str] (trian , live)
        :param output_shape:
        :return: X: [list, nd_array] : y: [list, int]
        """
        print(f"{'='*10} {'start transform':^2} {'='*10} ")
        print(f" mode= {mode} ")

        if mode == 'train':

            total_frame = [[self._video_to_frame(name) for name in fl] for fl in self.file_list]

            labels = [[[l] for _ in fl] for fl, l in zip(total_frame, self.labels)]

            X = torch.vstack(list(chain.from_iterable(total_frame)))
            y = torch.Tensor(labels).squeeze(1)

            print(f'video to frame done || total {X.shape}')
            print(f'video to frame done || total {y.shape}')
            print(f"{'='*10} {'end transform':^2} {'='*10}")
            return X, y

        elif mode == 'extract':
            import torchvision.models as models
            import torch.nn as nn

            resnet_50 = models.resnet50(pretrained=True)
            resnet_50.fc = nn.Linear(resnet_50.fc.in_features, 200)
            for param in resnet_50.parameters():
                param.requires_grad_(False)

            total_frame = []
            for class_file in self.file_list:
                total_frame.append([self.__video_to_frame(name, model=resnet_50) for name in class_file])

            # labels = [len(fl) * [l] for fl, l in zip(total_frame, self.labels)]
            labels = [[[l]for _ in fl] for fl, l in zip(total_frame, self.labels)]

            X = torch.stack(list(chain.from_iterable(total_frame)))
            print(f'torch X {X.shape}')
            y = torch.Tensor(labels).squeeze(1)
            print(f'torch y {y.shape}')

            print(f"{'=' * 10} {'end transform':^2} {'=' * 10}")
            return X, y

        else:
            """
            not working yet temp!
            """
            # live fit
            print(f'not implemented')
            return None

    def video_loader(self):
        """
        :return: generator
        """
        X, y = self.make_frame(mode='train')

        for gen in zip(X, y):
            yield gen
