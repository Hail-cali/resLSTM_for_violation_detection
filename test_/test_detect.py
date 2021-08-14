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
print([(frames[0].shape,frames[-1].shape) for frames in total_frame])
