#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 19:04:55 2020

@author: aktasos
"""
import itertools
import numpy as np
import pandas as pd
import pandas as pd
import os 
import json
import matplotlib.pyplot as plt
from matplotlib import colors
from pathlib import Path
from torch.utils.data import Dataset
from collections import OrderedDict, Counter, defaultdict

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


# def data_process(task_list):
    

# # class ARCDataset(dataset):
    
# #     def __init__(self, task_list):
        
    
class ARCParameters():
    def __init__(self, train_tasks): # task = train_tasks[x]          ['train'][0]['input']
        self.tasks = train_tasks
        
        self.params_analysis_list = []
        self.results = None
        
    def analyse_parameters(self):
        
        for task in self.tasks:
          
            task_n = TaskParameters(task)
            task_n.train_params()
            task_n.compare_train()
            task_n.count_good_params()
            
            self.params_analysis_list.append(task_n.good_params)
            res = np.array(self.params_analysis_list)
        self.results = np.sum(res, 0)
        
    # def results(self):
    #     for a, b in itertools.combinations(self.params_analysis_list, 2):
    #         print(a==b)
    #     for l in self.params_analysis_list:
            

class TaskParameters(): 
    
    def __init__(self, task): # task = train_tasks[x]          ['train'][0]['input']
        self.task = task
        self.nb_of_train = len(task['train'])
        self.train_list = []
        self.params_comparison = []
        self.good_params_count = []
        self.good_params = []
        self.nb_of_params = int()
            
    def train_params(self):
        for train in self.task['train']:
            data = TrainParameters(train)
            data.colors_params()
            # self.train_list.append(data.__dict__)  ##bring all attributes in a dictionary
            self.train_list.append(data.train_params_dict)
        # self.params_comparison = self.train_list[0].fromkeys(self.train_list[0],[])
        # self.good_params_count = self.train_list[0].fromkeys(self.train_list[0],0)    
        self.nb_of_params = len(data.train_params_dict)
        self.good_params_count = [0]*self.nb_of_params
        self.good_params = [0]*self.nb_of_params
        
        
    def compare_train(self):     
        n=0
        for a, b in itertools.combinations(self.train_list, 2):
            self.params_comparison.append([])
            n += 1
            for key in a.keys():
               
                compare = (a[key] == b[key])
                self.params_comparison[n-1].append(int(compare))
               
                
                
    def count_good_params(self):
    
        for n in range(len(self.params_comparison)) :
            for i in range(self.nb_of_params):
                self.good_params_count[i] += self.params_comparison[n][i]
            
        good = np.where(np.array(self.good_params_count) == len(self.params_comparison))
        for g in np.nditer(good):
        
            self.good_params[int(g)] = 1 
            
                
               
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
        
        
        
class TrainParameters(): # train_data = train_tasks[0]['train'][0]       ['input']
    
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
        self.train_params_dict = OrderedDict([])
        self.train_params = [self.x_in, 
                             self.y_in, 
                             self.x_out, 
                             self.y_out, 
                             self.ratio_in, 
                             self.ratio_out]
        
    def colors_params(self):
        no_color = [0]*9
        for key, data in self.train_data.items():
            
            if key == 'input':
                self.train_params_dict['x_in'] = self.x_in
                self.train_params_dict['y_in'] = self.y_in
                self.train_params_dict['ratio_in'] = self.ratio_in
            else : 
                self.train_params_dict['x_out'] = self.x_out
                self.train_params_dict['y_out'] = self.y_out
                self.train_params_dict['ratio_out'] = self.ratio_out
                
            data = np.array(data)    
            
            self.train_params_dict[key+"_stdev"] = (np.std(data))
            self.train_params.append(np.std(data))
            
            self.color_distrib['std_x'] = np.std(data,axis=0)
            self.color_distrib['std_y'] = np.std(data,axis=1)
            
            self.train_params.append(np.var(data))
            self.color_distrib['var_x'] = np.var(data,axis=0)
            self.color_distrib['var_y'] = np.var(data,axis=1)
            
            for i, ele in self.color_distrib.items():
            
                self.train_params_dict[f"{key}_{i}_median"] = np.median(ele)
                self.train_params.append(np.median(ele))
                
                self.train_params_dict[f"{key}_{i}_mean"] = np.mean(ele)
                self.train_params.append(np.mean(ele))
                
                for quantile in np.linspace(0.1, 1, 11):
                    
                    self.train_params_dict[f"{key}_{i}_quantile_{quantile}"] = np.quantile(ele, quantile)
                    self.train_params.append(np.quantile(ele, quantile))
                
                
            for color in range(10): 
                case_by_color = (np.count_nonzero(data==color))
                if case_by_color == 0:  
                    self.train_params_dict[f"{key}_color_{color}"] = [color] + no_color
                    self.train_params.extend(no_color)
                else:
                    x,y = np.where(data==color)
                    
                    self.train_params_dict[f"{key}_color_{color}"] = OrderedDict([
                                                                                  ('color', color), 
                                                                                  ('case_by_color', case_by_color), 
                                                                                  ('stdev_x', np.std(x)), 
                                                                                  ('stdev_y', np.std(y)), 
                                                                                  ('var_x', np.var(x)), 
                                                                                  ('var-y', np.var(y)),
                                                                                  ('mean_x', np.mean(x)), 
                                                                                  ('mean_y', np.mean(y)),
                                                                                  ('median_x', np.median(x)), 
                                                                                  ('median_y', np.median(y))
                                                                                  # ('ratio_stdev', np.std(x)/np.std(y)), 
                                                                                  # ('ratio_var', np.var(x)/np.var(y)), 
                                                                                  # ('ratio_stdev_var_x', np.std(x)/np.var(x)), 
                                                                                  # ('ratio_stdev_var_y', np.std(y)/np.var(y)) 
                                                                                  ])
                    self.train_params.extend((color,
                                              case_by_color, 
                                              np.std(x), 
                                              np.std(y), 
                                              np.var(x), 
                                              np.var(y), 
                                              # np.std(x)/np.std(y), 
                                              # np.var(x)/np.var(y), 
                                              # np.std(x)/np.var(x), 
                                              # np.std(y)/np.var(y), 
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
    
          




# first=TaskParameters(train_tasks[0])  
# first.train_params()
# first.compare_train()
# first.count_good_params()

results = ARCParameters(train_tasks)
results.analyse_parameters()





