import pandas as pd
import numpy as np
from utils.data_loader import *
from model.feature_net import *

DPATH = '../data/fight'


loader = DataLoader(path=DPATH)
print(loader.file_list)
total_frame = loader.make_frame(mode='train')

print(len(total_frame))
print([len(frames) for frames in total_frame])
print([frames[-1] for frames in total_frame])

model = FeatureNet()


sample_file_list = total_frame[1:3]
total_feature_map = model.transfrom_video(sample_file_list)

#total_feature_map = model.transfrom_video(total_frame)

print(f'len : {len(total_feature_map)}')


