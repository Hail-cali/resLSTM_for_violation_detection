import pandas as pd
import numpy as np
import os
import cv2



class DataLoader(object):

    def __init__(self, path=None, labels=[], batch_size=64, mode='train', img_resize=False, verbose=False):
        self.path = path
        self.labels = labels
        self.batch_size = batch_size
        self.mode = mode
        self.verbose = verbose
        self.img_resize = img_resize
        self.total_frame = list()

        try:
            self.file_list = os.listdir(self.path)
        except FileNotFoundError:
            print(f'[Errno 2] No such file or directory: {self.path}')
            self.file_list = None

        self.data_size = len(self.file_list)

    def _video_to_frame(self, file_name):
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

        if len(frames) > 80:

            return frames[:80]
        elif len(frames) < 80:
            zero_padding = np.zeros(frames[0].shape)
            # print(f'zero padding len {len([zero_padding]*(80-len(frames)))}')
            # print(f'video frames len {len(frames + ([zero_padding]*(80-len(frames))))}')
            return frames + ([zero_padding]*(80-len(frames)))
        else:
            return frames


    def make_frame(self, mode='train', output_shape=(360, 640, 3), verbose=False):
        """
        :param verbose:
        :param mode: [str] (trian , live)
        :param output_shape:
        :return: [list, list]
        """
        print(f"{'='*10} {'start transform':^2} {'='*10} ")
        if mode == 'train':

            total_frame = [self._video_to_frame(name) for name in self.file_list]

            # 이런식으로 작동하는거 cash로 돌렸는데 batch size 설정안하면 데이터 크기에 따라 메모리 부족날 수도
            # total_frame = []
            # for name in self.file_list[:2]:
            #     frames = self._video_to_frame(name)
            #     total_frame.append(frames)
            print(f'video to frame done || total {len(total_frame)}  shape {total_frame[0][0].shape}')
            return total_frame

        else:
            """
            not working yet temp!
            """

            # live fit
            model = 'some model'  # temp
            total_feature_frame = []
            for batch in range(0, self.data_size, self.batch_size):
                batch_frame = [self._video_to_frame(name) for name in self.file_list[:batch]]
                features = model.fit(batch_frame)
                total_feature_frame.append(features)
            # for name in self.file_list[:2]:
            #     filepath = os.path.join(self.path, name)
            #     #models.method => feature map
            #     frames = self._video_to_frame(filepath)
            #     self.total_frame.append(frames)
            return total_feature_frame
