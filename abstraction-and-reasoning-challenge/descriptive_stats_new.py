#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 19:04:55 2020

@author: aktasos
"""

import numpy as np
from collections import OrderedDict
import torch
import itertools

def detect_borders(matrix):
    transp_matrix = matrix.transpose(0, 1)
    up, u_count = matrix[0].unique(return_counts=True)
    right, r_count = transp_matrix[0].unique(return_counts=True)
    down, d_count = matrix[-1].unique(return_counts=True)
    left, l_count = transp_matrix[-1].unique(return_counts=True)
    values = torch.cat((up, right, down, left))
    unique_values = values.unique()
    count = torch.cat((u_count, r_count, d_count, l_count))
    border_dict = {}
    for uv in unique_values:
        border_dict[int(uv)] = 0
        indices = np.where(values == int(uv))[0]
        for i in indices:
            border_dict[int(uv)] += int(count[i])
    max_color = max(border_dict, key=border_dict.get)
    if border_dict[max_color] < sum(matrix.size()):
        color = -1
    else:
        color = max_color
    return color

def detect_grids(matrix):
    lines = []
    columns = []
    grid = False
    for index, line in enumerate(matrix):
        unique_values = line.unique()
        if len(unique_values) == 1:
            lines.append([index, int(unique_values)])
    for index, column in enumerate(matrix.transpose(0, 1)):
        unique_values = column.unique()
        if len(unique_values) == 1:
            columns.append([index, int(unique_values)])
    if lines and columns:
        grid = True
    return lines, columns, grid

def background(matrix):
    if 0 in matrix.unique():
        bg = 0
    else:
        bg = detect_borders(matrix)
    return bg

class TaskParameters():
    """class with all features(params) in task"""
    def __init__(self, task, task_index): # task = train_tasks[x]          ['train'][0]['input']
        self.task = task
        self.task_index = task_index
        self.nb_of_train = len(task['train'])
        self.train_dict_list = []
        self.params_comparison = OrderedDict()
        self.good_params_count = []
        self.good_params = OrderedDict()

    def train_params(self):
        """extract all features(params) in task"""
        for train in self.task['train']:
            data = TrainParameters(train, self.task_index)
            data.basic_params()
            data.colors_params()
            data.colors_in_out()
            data.others()
            self.train_dict_list.append(data.params)

    def compare_train(self):
        """compare features(params) in task to look if they change"""
        for key in self.train_dict_list[0]:
            k = []
            for train in self.train_dict_list:
                try:
                    k.append(train[key])
                except:
                    break
            try:
                val = set(k)
            except TypeError:
                try:
                    val = np.unique(k)
                except:
                    pass
            if len(k) == len(self.train_dict_list) and len(val) == 1:
                self.good_params[key] = k[0]

class TrainParameters(): # train_data = train_tasks[0]['train'][0]       ['input']
    """class with all features(params) in one example"""
    def __init__(self, train_data, task_index):
        self.train_data = train_data
        self.task_index = task_index
        self.input = np.array(train_data['input'])
        self.output = np.array(train_data['output'])
        self.torch_input = torch.tensor(train_data['input'])
        self.torch_output = torch.tensor(train_data['output'])
        self.params = OrderedDict([])
        self.train_params = []
        self.params['nb_of_color_in'] = len(set(list(itertools.chain(*train_data['input']))))
        self.params['nb_of_color_out'] = len(set(list(itertools.chain(*train_data['output']))))
        if self.params['nb_of_color_out'] == 1:
            self.params['single_color_out'] = 1
            self.params['color_out'] = set(list(itertools.chain(*train_data['output']))).pop()
        if self.params['nb_of_color_out'] == 2:
            self.params['double_color_out'] = 1
            self.params['color_out1'] = set(list(itertools.chain(*train_data['output']))).pop()
            self.params['color_out2'] = set(list(itertools.chain(*train_data['output']))).pop()

    def basic_params(self):
        self.params['input'] = float('0.'+(''.join(str(n) for l in self.train_data['input'] for n in l)))
        self.params['output'] = float('0.'+(''.join(str(n) for l in self.train_data['output'] for n in l)))
        self.params['x_in'] = len(self.train_data['input'][0])
        self.params['y_in'] = len(self.train_data['input'])
        self.params['x_out'] = len(self.train_data['output'][0])
        self.params['y_out'] = len(self.train_data['output'])
        self.params['ratio_in'] = self.params['x_in'] / self.params['y_in']
        self.params['ratio_out'] = self.params['x_out'] / self.params['y_out']
        self.params['ratio_x'] = self.params['x_in'] / self.params['x_out']
        self.params['ratio_y'] = self.params['y_in'] / self.params['y_out']
        self.params['nb_of_case_in'] = self.params['x_in'] * self.params['y_in']
        self.params['nb_of_case_out'] = self.params['x_out'] * self.params['y_out']

    def colors_params(self):
        """extract features(params) in one example"""
        for key, data in self.train_data.items():
            max_color_case = 0
            self.params[f'{key}_task_index'] = self.task_index
            if key == 'input':
                nb_of_case = self.params['x_in'] * self.params['y_in']
            else:
                nb_of_case = self.params['x_out'] * self.params['y_out']
            data = np.array(data)
            self.params[f"{key}_stdev"] = (np.std(data))
            for idx, ele in enumerate(np.std(data, axis=0)):
                self.params[f'{key}_color_distrib_std_x_{idx}'] = ele
            for idx, ele in enumerate(np.std(data, axis=1)):
                self.params[f'{key}_color_distrib_std_y_{idx}'] = ele
            for idx, ele in enumerate(np.var(data, axis=0)):
                self.params[f'{key}_color_distrib_var_x_{idx}'] = ele
            for idx, ele in enumerate(np.var(data, axis=1)):
                self.params[f'{key}_color_distrib_var_y_{idx}'] = ele
            for color in range(10):
                case_by_color = (np.count_nonzero(data == color))
                if color > 0 and case_by_color > max_color_case:
                    max_color_case = case_by_color
                    x, y = np.where(data == color)
                    self.params.update({
                        f"{key}_max_color": color,
                        f'{key}_max_color_case_by_color': case_by_color,
                        f'{key}_max_color_proportion': case_by_color/(nb_of_case),
                        f'{key}_max_color_stdev_x': np.std(x),
                        f'{key}_max_color_stdev_y': np.std(y),
                        f'{key}_max_color_var_x': np.var(x),
                        f'{key}_max_color_var-y': np.var(y),
                        f'{key}_max_color_mean_x': np.mean(x),
                        f'{key}_max_color_mean_y': np.mean(y),
                        f'{key}_max_color_median_x': np.median(x),
                        f'{key}_max_color_median_y': np.median(y)
                        })
                if case_by_color == 0:
                    self.params.update({
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
                    x, y = np.where(data == color)
                    self.params.update({
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

    def colors_in_out(self):
        try:
            self.params["input_max_color"] and self.params["output_max_color"]
            self.params['ratio_max_color'] = self.params["input_max_color"]/self.params["output_max_color"]
            self.params['ratio_max_color_case_by_color'] = self.params['input_max_color_case_by_color']/self.params['output_max_color_case_by_color']
            self.params['ratio_max_color_proportion'] = self.params['input_max_color_proportion']/self.params['output_max_color_proportion']
            self.params['ratio_max_color_stdev_x'] = self.params['input_max_color_stdev_x']/self.params['output_max_color_stdev_x']
            self.params['ratio_max_color_stdev_y'] = self.params['input_max_color_stdev_y']/self.params['output_max_color_stdev_y']
            self.params['ratio_max_color_var_x'] = self.params['input_max_color_var_x']/self.params['output_max_color_var_x']
            self.params['ratio_max_color_var-y'] = self.params['input_max_color_var-y']/self.params['output_max_color_var-y']
            self.params['ratio_max_color_mean_x'] = self.params['input_max_color_mean_x']/self.params['output_max_color_mean_x']
            self.params['ratio_max_color_mean_y'] = self.params['input_max_color_mean_y']/self.params['output_max_color_mean_y']
            self.params['ratio_max_color_median_x'] = self.params['input_max_color_median_x']/self.params['output_max_color_median_x']
            self.params['ratio_max_color_median_y'] = self.params['input_max_color_median_y']/self.params['output_max_color_median_y']
        except:
            pass

    def others(self):
        self.params['input_border_color'] = detect_borders(self.torch_input)
        self.params['input_background_color'] = background(self.torch_input)
        lines, columns, grid = detect_grids(self.torch_input)
        l = 0
        c = 0
        self.params[f'input_nb_of_lines'] = len(lines)
        self.params[f'input_nb_of_columns'] = len(columns)
        for line in lines:
            self.params[f'input_line_{l}_index'], self.params[f'input_line_{l}_color'] = line
            l += 1
        for column in columns:
            self.params[f'input_column_{c}_index'], self.params[f'input_column_{c}_color'] = column
            c += 1
        self.params['output_border_color'] = detect_borders(self.torch_output)
        self.params['output_background_color'] = background(self.torch_output)
        lines, columns, grid = detect_grids(self.torch_output)
        l = 0
        c = 0
        self.params[f'output_nb_of_lines'] = len(lines)
        self.params[f'output_nb_of_columns'] = len(columns)
        for line in lines:
            self.params[f'output_line_{l}_index'], self.params[f'output_line_{l}_color'] = line
            l += 1
        for column in columns:
            self.params[f'output_column_{c}_index'], self.params[f'output_column_{c}_color'] = column
            c += 1
