#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 19:04:55 2020

@author: aktasos
"""
import itertools
import numpy as np


import os 
import json

from pathlib import Path
from torch.utils.data import Dataset
from collections import OrderedDict
from descriptive_stats import ARCParameters
path = os.path.dirname(os.path.abspath(__file__))
for dirname, _, filenames in os.walk(path):
        print(dirname)
        


data_path = Path(path)
training_path = data_path / 'training'
evaluation_path = data_path / 'evaluation'
test_path = data_path / 'test'
training_tasks = sorted(os.listdir(training_path))



def data_openner(tasks, path):
    task_list =[]
    for task in tasks :
        task_file = str(path / task)
        with open(task_file, 'r') as f:
            task_list.append(json.load(f))
            
    return task_list



train_tasks = data_openner(training_tasks, training_path)      


# first=TaskParameters(train_tasks[0])  
# first.train_params()
# first.compare_train()
# first.count_good_params()

results = ARCParameters(train_tasks)
results.analyse_parameters()





