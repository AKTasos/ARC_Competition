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
from torch.utils.data import Dataset
from collections import OrderedDict

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



train_task = data_openner(training_tasks, training_path)      


# def data_process(task_list):
    

# # class ARCDataset(dataset):
    
# #     def __init__(self, task_list):
        
    


     
class TaskParameters(): 
    
    def __init__(self, task): # task = train_task[x]          ['train'][0]['input']
        self.task = task
        self.nb_of_train = len(task['train'])
        self.train_list = []
    
            
    def train_params(self):
        for train in self.task['train']:
            data = TrainParameters(train)
            data.colors_params()
            # self.train_list.append(data.__dict__)  ##bring all attributes in a dictionary
            self.train_list.append(data.color_params)
               
    # def size_rules(self):
    #     self.train_list
        
    #     self.train_data = train_data
    #     self.x_in = len(train_data['input'][0])
    #     self.x_out = len(train_data['output'][0])
    #     self.y_in = len(train_data['input'])
    #     self.y_out = len(train_data['output'])
    #     self.ratio_in = self.x_in / self.y_in
    #     self.ratio_out =self.x_out / self.y_out
        
    # def colors_rules(self):
        
    #     self.colors_in = None
    #     self.colors_out = None
    #     self.nb_of_colors_in = None
    #     self.nb_of_colors_out = None
    #     self.case_by_color_in = str()
    #     self.case_by_color_out = str()
        
        
        
class TrainParameters(): # train_data = train_task[0]['train'][0]       ['input']
    
    def __init__(self, train_data):
        self.train_data = (train_data)
        self.input = np.array(train_data['input'])
        self.output =np.array(train_data['output'])
    
        self.x_in = len(train_data['input'][0])
        self.x_out = len(train_data['output'][0])
        self.y_in = len(train_data['input'])
        self.y_out = len(train_data['output'])
        self.ratio_in = self.x_in / self.y_in
        self.ratio_out =self.x_out / self.y_out
        self.case_by_color_in = []
        self.case_by_color_out = []
        self.color_distrib = OrderedDict()
        self.train_params_names = x_in, y_in, x_out, y_out, ratio_in, ratio_out
        self.train_params = [self.x_in, self.y_in, self.x_out, self.y_out, self.ratio_in, self.ratio_out]
        
    def colors_params(self):
        no_color = [0]*14
        for data in self.train_data.values():
            data = np.array(data)    
            
            
            self.train_params.append(np.std(data))
            self.color_distrib['std_x'] = np.std(data,axis=0)
            self.color_distrib['std_y'] = np.std(data,axis=1)
            
            self.train_params.append(np.var(data))
            self.color_distrib['var_x'] = np.var(data,axis=0)
            self.color_distrib['var_y'] = np.var(data,axis=1)
            
            for ele in self.color_distrib.values():
            
                self.train_params.append(np.median(ele))
                self.train_params.append(np.mean(ele))
                
                for quantile in np.linspace(0, 1, 11):
                    self.train_params.append(np.quantile(ele, quantile))
                
                
            for color in range(10): 
                case_by_color = (np.count_nonzero(data==color))
                if case_by_color == 0:  
                    self.train_params.extend(no_color)
                else:
                    x,y=np.where(data==color)
                    self.train_params.extend((color,
                                              case_by_color, 
                                              np.std(x), 
                                              np.std(y), 
                                              np.var(x), 
                                              np.var(y), 
                                              np.std(x)/np.std(y), 
                                              np.var(x)/np.var(y), 
                                              np.std(x)/np.var(x), 
                                              np.std(y)/np.var(y), 
                                              np.mean(x), 
                                              np.mean(y), 
                                              np.median(x), 
                                              np.median(y)))
                
                
                
                
                
                
                
                # self.case_by_color_out.append(np.count_nonzero(self.output==color))
                # if self.case_by_color_out[color] == 0:  
                #     self.train_params.extend(no_color)
                # else:
                #     x,y=np.where(self.output==color)
                #     self.train_params.extend((color, 
                #                               self.case_by_color_out[color], 
                #                               np.std(x), 
                #                               np.std(y), 
                #                               np.var(x), 
                #                               np.var(y), 
                #                               np.std(x)/np.std(y), 
                #                               np.var(x)/np.var(y), 
                #                               np.std(x)/np.var(x), 
                #                               np.std(y)/np.var(y), 
                #                               np.mean(x), 
                #                               np.mean(y), 
                #                               np.median(x), 
                #                               np.median(y)))
    
          

  
        