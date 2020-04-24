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
import json
import itertools
from collections import OrderedDict
from pathlib import Path
import matplotlib.pyplot as plt

PATH = os.path.dirname(os.path.abspath(__file__))

data_path = Path(PATH)
training_path = data_path / 'training'
evaluation_path = data_path / 'evaluation'
test_path = data_path / 'test'
training_tasks = sorted(os.listdir(training_path))

data_path = "./training_results/params_data.json"

def data_openner(tasks, path):
    
    """open and load json files"""
    task_list = []
    for task in tasks:
        task_file = str(path / task)
        with open(task_file, 'r') as f:
            task_list.append(json.load(f))
    return task_list

train_tasks = data_openner(training_tasks, training_path)


class TasksTensorDataset(Dataset): 
    # train_tasks = train_tasks[x]
    def __init__(self, task):
        self.task = task
        self.y = []
        self.x = []
        self.n_samples = len(self.task['train'])
        for train in self.task['train']:
            self.x.append(train['input'])
            self.y.append(train['output'])
        self.x_torch = np.heaviside(torch.LongTensor(self.x).view(self.n_samples,1,3,3),0)
        self.y_array = np.heaviside(self.y, 0)
        self.y_torch = torch.LongTensor(self.y_array).view(self.n_samples,9,9)


    def __getitem__(self, index):
        try:
            return self.x_torch[index], self.y_torch[index]
        except:
            return self.x_torch[index]
        
    def __len__(self):
        return self.n_samples



class FeaturesTensorDataset(Dataset):

    def __init__(self, data_path, task_nb=-1):
        data = pd.read_json(data_path)
        # data = np.loadtxt("./input/data.csv", delimiter=',', dtype=np.float32, skiprows=1)
        try:
            self.y = data.iloc[1::2]
            self.y_torch = torch.FloatTensor(np.array(data.iloc[1::2]))
        except:
            self.y = None
            self.y_torch = None

        self.x = data.iloc[::2]
        self.x_torch = torch.FloatTensor(np.array(data.iloc[::2]))

        self.n_samples = len(data[1::2])
        self.names = data.columns
        self.task_nb = task_nb
        self.y_task = None
        self.x_task = None
        if self.task_nb != -1:
            self.task_data()


    def task_data(self):

        y_task_t = self.y['input_task_index'] == self.task_nb
        x_task_t = self.x['input_task_index'] == self.task_nb
        self.y_task = torch.FloatTensor(np.array(self.y[y_task_t]))
        self.x_task = torch.FloatTensor(np.array(self.x[x_task_t]))
        self.n_samples = len(self.x_task)

    def __getitem__(self, index):
        if self.task_nb != -1:
            try:
                return self.x_task[index], self.y_task[index]
            except:
                return self.x_task[index]
        else:
            try:
                return self.x_torch[index], self.y_torch[index]
            except:
                return self.x_torch[index]

    def __len__(self):
        return self.n_samples


    