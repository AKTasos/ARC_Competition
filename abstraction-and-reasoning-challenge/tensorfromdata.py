#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 14:23:09 2020

@author: aktasos
"""
from torch.utils.data import Dataset
import os
import pandas as pd
import torch
import numpy as np

os.path.dirname(os.path.abspath(__file__))
print(os.listdir())
data_path = "./training_results/params_data.json"


class TensorDataset(Dataset,task=None):
    
    def __init__(self, data_path) : 
        data = pd.read_json(data_path)
        # data = np.loadtxt("./input/data.csv", delimiter=',', dtype=np.float32, skiprows=1)
        try :
            self.y = torch.FloatTensor(np.array(data.iloc[1::2]))
        except :
            self.y = None
        self.x = torch.FloatTensor(np.array(data.iloc[::2]))
        self.n_samples=len(data[1::2])
        self.names = data.columns
        self.task = task
        
    def task_data(self)    
        if self.task != None:
        
        
    def __getitem__(self, index):
        try :
            return self.x[index], self.y[index]
        except :
            return self.x[index]
        
    
    def __len__(self):
        return self.n_samples