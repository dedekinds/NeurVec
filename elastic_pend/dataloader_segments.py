from torch.utils import data
import os
import numpy as np
import time
import glob

class Dataset_training(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data_dir, train_T, seg_num=100):

        self.data = np.load(data_dir) # T*bs*dim
        idx = np.arange(self.data.shape[1]).tolist()
        self.data = self.data[:, idx]
        self.seg_num = seg_num
        self.size = self.data.shape[1]*self.seg_num//1
        self.train_T = train_T
        print(self.data.shape)
    def __len__(self):
        'Denotes the total number of samples'
        return self.size

    def __getitem__(self, index):
        'Generates one sample of data'
        batch_idx = index // (self.seg_num//1)
        seg_idx = index % (self.seg_num//1)
        # print(batch_idx, seg_idx)
        sample = self.data[seg_idx*1:seg_idx*1+self.train_T, batch_idx, :]
        return sample

class Dataset_testing(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data_dir):
        self.data = np.load(data_dir) # T*bs*dim

        idx = np.arange(self.data.shape[1]).tolist()
        self.data = self.data[:, idx]

        self.size = self.data.shape[1]

    def __len__(self):
        'Denotes the total number of samples'
        return self.size

    def __getitem__(self, index):
        'Generates one sample of data'
        sample = self.data[:,index,:]
        return sample