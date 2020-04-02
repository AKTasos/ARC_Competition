#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 19:04:55 2020

@author: aktasos
"""

import numpy as np
import pandas as pd
import pandas as pd
import os 
import json
import matplotlib.pyplot as plt
from matplotlib import colors
from pathlib import Path

path = os.path.dirname(os.path.abspath(__file__))
for dirname, _, filenames in os.walk(path):
        print(dirname)
        


data_path = Path(path)
training_path = data_path / 'training'
evaluation_path = data_path / 'evaluation'
test_path = data_path / 'test'
training_tasks = sorted(os.listdir(training_path))



def dataloader(training_tasks):
    train_task =[]
    for task in training_tasks :
        train_task_file = str(training_path / task)
        with open(train_task_file, 'r') as f:
            train_task.append(json.load(f))
            
    return train_task



train_task = dataloader(training_tasks)      



     
class TaskParameters(): 
    
    def __init__(self, task): # task = train_task[x]          ['train'][0]['input']
        self.task = task
        self.nb_of_train = len(task['train'])
        self.train_dict = {}
        
        
    def train_params(self):
        
        n=97
        for train in self.task['train']:
            data = TrainParameters(train)
            # self.train_dict[chr(n)] = data.__dict__
            n += 1
               
    def size_rules(self):
        print('a') 
    def colors_rules(self):
        print('b')
        
        
        
class TrainParameters(): # train_data = train_task[0]['train'][0]       ['input']
    
    def __init__(self, train_data):
        self.train_data = train_data
        self.x_in = len(train_data['input'][0])
        self.x_out = len(train_data['output'][0])
        self.y_in = len(train_data['input'])
        self.y_out = len(train_data['output'])
        self.ratio_in = self.x_in / self.y_in
        self.ratio_out =self.x_out / self.y_out
        self.colors_in = None
        self.colors_out = None
        self.nb_of_colors_in = None
        self.nb_of_colors_out = None
        self.case_by_color_in = str()
        self.case_by_color_out = str()
        
        
    def colors_params(self):
        flat_in = [item for sublist in self.train_data['input'] for item in sublist]
        flat_out = [item for sublist in self.train_data['output'] for item in sublist]
        
        self.colors_in = ''.join(map(str, sorted(set(flat_in))))
        self.colors_out = ''.join(map(str, sorted(set(flat_in))))
        
        self.nb_of_colors_in = len(set(flat_in))
        self.nb_of_colors_out = len(set(flat_out))
        
        for x in range(10): 
            self.case_by_color_in += str(flat_in.count(x))
            self.case_by_color_out += str(flat_out.count(x))
            
    # def return_parameters(self):
        
        
            # flat_in.count(x) = ''.join(map(str, flat_in)))
            # flat_out.count(x) = ''.join(map(str, flat_out)))
            
            
        