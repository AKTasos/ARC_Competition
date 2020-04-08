#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 14:23:09 2020

@author: aktasos
"""
from torch.utils.data import Dataset
import os

import torch


os.path.dirname(os.path.abspath(__file__))
print(os.listdir())
data_path = str


class TensorDataset(Dataset):
    
    def __init__(self, data) : 
        # data = pd.read_csv(data_path)
        # data = np.loadtxt("./input/data.csv", delimiter=',', dtype=np.float32, skiprows=1)
        try :
            self.y = torch.from_numpy(data[1::2])
        except :
            self.y = None
        self.x = torch.from_numpy(data[::2])
        self.n_samples=data.shape[0]
        
    def __getitem__(self, index):
        try :
            return self.x[index], self.y[index]
        except :
            return self.x[index]
        
    
    def __len__(self):
        return self.n_samples