#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 19:04:55 2020

@author: aktasos
"""
import json
import itertools
import numpy as np
from collections import OrderedDict
import os
from pathlib import Path
import torch
import pandas as pd
import matplotlib.pyplot as plt
PATH = os.path.dirname(os.path.abspath(__file__))
for dirname, _, filenames in os.walk(PATH):
    print(dirname)


data_path = Path(PATH)
training_path = data_path / 'training'
evaluation_path = data_path / 'evaluation'
test_path = data_path / 'test'
training_tasks = sorted(os.listdir(training_path))

def correct(preds, labels):
    c=torch.eq(preds.argmax(dim=3),labels)
    return c.sum().item()

def correct_output(preds, labels):
    c=torch.eq(preds,labels)
    return c.sum().item()

def data_openner(tasks, path):
    """open and load json files"""
    task_list = []
    for task in tasks:
        task_file = str(path / task)
        with open(task_file, 'r') as f:
            task_list.append(json.load(f))
    return task_list

class ARCParameters():
    """class with all features(params) in ARC dataset"""
    def __init__(self, train_tasks):  # task = train_tasks[x]          ['train'][0]['input']    
        self.tasks = train_tasks
        self.current_task = 0
        self.params_analysis_list = []
        self.params_analysis_results = None
        self.params_data = []
        self.params_labels = []

    def analyse_parameters(self):
        """extract all features(params) in ARC dataset"""
        n = 0
        for task in self.tasks:
            n += 1
            print(n)
            task_n = TaskParameters(task, self.current_task)
            task_n.train_params()
            task_n.compare_train()
            task_n.count_good_params()
            # self.params_data.append(task_n.train_list)
            self.current_task += 1
            self.params_analysis_list.append(task_n.good_params)
            res = np.array(self.params_analysis_list)
            self.params_data += [train for train in task_n.train_list]

        self.params_analysis_results = np.sum(res, 0)
        self.params_labels = task_n.params_labels
    # def results(self):
    #     for a, b in itertools.combinations(self.params_analysis_list, 2):
    #         print(a==b)
    #     for l in self.params_analysis_list:

    def save(self):
        """save all features(params) in json file"""
        data = pd.DataFrame(self.params_data, columns=self.params_labels)
        data.to_json("./training_results/params_data.json", orient='columns')


class TaskParameters():
    """class with all features(params) in task"""
    def __init__(self, task, task_index): # task = train_tasks[x]          ['train'][0]['input']
        self.task = task
        self.task_index = task_index
        self.nb_of_train = len(task['train'])
        self.train_dict_list = []
        self.train_list = []
        self.params_comparison = []
        self.good_params_count = []
        self.good_params = []
        self.nb_of_params = int()
        self.params_labels = []

    def train_params(self):
        """extract all features(params) in task"""
        for train in self.task['train']:
            data = TrainParameters(train, self.task_index)
            data.colors_params()
            # self.train_list.append(data.__dict__)  ##bring all attributes in a dictionary
            self.train_dict_list.append(data.train_params_dict)
            self.train_list += data.train_params
        # self.params_comparison = self.train_list[0].fromkeys(self.train_list[0],[])
        # self.good_params_count = self.train_list[0].fromkeys(self.train_list[0],0)    
        self.nb_of_params = len(data.train_params_dict)
        self.good_params_count = [0]*self.nb_of_params
        self.good_params = [0]*self.nb_of_params
        temp = [x for x in data.train_params_dict.keys()]
        self.params_labels = temp[:(len(temp)//2)]


    def compare_train(self):
        """compare features(params) in task to look if they change"""
        n = 0
        for a, b in itertools.combinations(self.train_dict_list, 2):
            self.params_comparison.append([])
            n += 1
            for key in a.keys():

                compare = (a[key] == b[key])
                self.params_comparison[n-1].append(int(compare))



    def count_good_params(self):
        """count features(params) in task that does not change"""
        for n in range(len(self.params_comparison)):
            for i in range(self.nb_of_params):
                self.good_params_count[i] += self.params_comparison[n][i]

        good = np.where(np.array(self.good_params_count) == len(self.params_comparison))
        for g in np.nditer(good):

            self.good_params[int(g)] = 1

    def plot_params(self):
        """plot features(params)"""
        
        for p in range(len(self.train_list[0])):
            plt.figure()
            plt.title(self.params_labels[p])
            x = []
            y = []
            x_array = np.array([])
            
           
            for n in range(self.nb_of_train):
                x.append(self.train_list[n*2][p])
                y.append(self.train_list[n*2+1][p])
                
                x_array = np.append(x_array, [1, self.train_list[n*2][p], (self.train_list[n*2][p])**2])
                
                
            # Normal Equation
            y_array = np.array(y)
            x_array = x_array.reshape(len(x_array)//3,3)
            theta = np.linalg.pinv((np.transpose(x_array) @ x_array)) @ (np.transpose(x_array) @ y_array)
            plt.scatter(x, y)
            x_function = u=np.arange(min(x), max(x), 0.1)
            y_function = theta[0] + (theta[1] * x_function) + (theta[2] * (x_function**2))
            plt.plot(x_function, y_function)
            plt.show()



class TrainParameters(): # train_data = train_tasks[0]['train'][0]       ['input']
    """class with all features(params) in one example"""
    def __init__(self, train_data, task_index):
        self.train_data = (train_data)
        self.task_index = task_index
        self.input = np.array(train_data['input'])
        self.output = np.array(train_data['output'])

        self.x_in = len(train_data['input'][0])
        self.x_out = len(train_data['output'][0])

        self.y_in = len(train_data['input'])
        self.y_out = len(train_data['output'])
        self.ratio_in = self.x_in / self.y_in
        self.ratio_out = self.x_out / self.y_out
        self.case_by_color_in = []
        self.case_by_color_out = []
        self.color_distrib = OrderedDict()
        self.train_params_dict = OrderedDict([])
        self.train_params = []

    def colors_params(self):
        """extract features(params) in one example"""
        # color_max = 0
        # nb_color_max = 0
        # color_min = 0
        # nb_color_min = 0
        for key, data in self.train_data.items():

            self.train_params_dict[f'{key}_task_index'] = self.task_index

            if key == 'input':

                self.train_params_dict['input'] = float('0.'+(''.join(str(n) for l in self.train_data['input'] for n in l)))
                self.train_params_dict['x_in'] = self.x_in
                self.train_params_dict['y_in'] = self.y_in
                self.train_params_dict['ratio_in'] = self.ratio_in
                nb_of_case = self.x_in * self.y_in
                nb_color_min = nb_of_case

            else:
                self.train_params_dict['output'] = float('0.'+(''.join(str(n) for l in self.train_data['output'] for n in l)))
                self.train_params_dict['x_out'] = self.x_out
                self.train_params_dict['y_out'] = self.y_out
                self.train_params_dict['ratio_out'] = self.ratio_out
                nb_of_case = self.x_out * self.y_out
                nb_color_min = nb_of_case

            data = np.array(data)

            self.train_params_dict[key+"_stdev"] = (np.std(data))
            self.color_distrib['std_x'] = np.std(data, axis=0)
            self.color_distrib['std_y'] = np.std(data, axis=1)
            self.color_distrib['var_x'] = np.var(data, axis=0)
            self.color_distrib['var_y'] = np.var(data, axis=1)

            for i, ele in self.color_distrib.items():
                self.train_params_dict[f"{key}_{i}_median"] = np.median(ele)
                self.train_params_dict[f"{key}_{i}_mean"] = np.mean(ele)

                for quantile in np.linspace(0.1, 1, 11):
                    self.train_params_dict[f"{key}_{i}_quantile_{quantile}"] = np.quantile(ele, quantile)

            for color in range(10):
                case_by_color = (np.count_nonzero(data == color))
                if case_by_color == 0:
                    self.train_params_dict.update({f"{key}_color_{color}": color,
                                                   f'{key}_color_{color}_case_by_color': 0,
                                                   f'{key}_color_{color}_proportion': 0,
                                                   f'{key}_color_{color}_stdev_x': 0,
                                                   f'{key}_color_{color}_stdev_y': 0,
                                                   f'{key}_color_{color}_var_x': 0,
                                                   f'{key}_color_{color}_var-y': 0,
                                                   f'{key}_color_{color}_mean_x': 0,
                                                   f'{key}_color_{color}_mean_y': 0,
                                                   f'{key}_color_{color}_median_x': 0,
                                                   f'{key}_color_{color}_median_y': 0
                                                   })
                else:
                    # if case_by_color > nb_color_max:
                    #     nb_color_max = case_by_color
                    # if case_by_color < nb_color_min:
                    #     nb_color_max = case_by_color
                    x, y = np.where(data == color)
                    self.train_params_dict.update({f"{key}_color_{color}": color,
                                                   f'{key}_color_{color}_case_by_color': case_by_color,
                                                   f'{key}_color_{color}_proportion': case_by_color/(nb_of_case),
                                                   f'{key}_color_{color}_stdev_x': np.std(x),
                                                   f'{key}_color_{color}_stdev_y': np.std(y),
                                                   f'{key}_color_{color}_var_x': np.var(x),
                                                   f'{key}_color_{color}_var-y': np.var(y),
                                                   f'{key}_color_{color}_mean_x': np.mean(x),
                                                   f'{key}_color_{color}_mean_y': np.mean(y),
                                                   f'{key}_color_{color}_median_x': np.median(x),
                                                   f'{key}_color_{color}_median_y': np.median(y)
                                                   })

        in_out = list(self.train_params_dict.values())   
        len(in_out)
        self.train_params.append(in_out[:len(in_out)//2])           
        self.train_params.append(in_out[len(in_out)//2:])          


# train_tasks = data_openner(training_tasks, training_path)

# arc_params = ARCParameters(train_tasks)
# arc_params.analyse_parameters()
# # # arc_params.save()
# first = TaskParameters(train_tasks[0], 0)
# first.train_params()
# first.compare_train()
# first.count_good_params()
# first.plot_params()
