#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 14:23:09 2020

@author: aktasos
"""
from torch.utils.data import Dataset
import torch
import numpy as np
import json
from descriptive_stats_new import TaskParameters
from tools import MatrixDecomposition

def data_openner(tasks, path):
    """open and load json files"""
    task_list = []
    for task in tasks:
        labels = task
        task_file = str(path / task)
        with open(task_file, 'r') as f:
            task_list.append((json.load(f), labels))
    return task_list

class AllTasksGroupedWithTest(Dataset):
    """prepare data for the dataloader"""

    def __init__(self, all_task):
        t = 0
        self.all_task = all_task
        self.dataset = []
        for task_info in self.all_task:
            task = task_info[0]
            self.task_id = task_info[1]
            feat = TaskParameters(task, t)
            feat.train_params()
            feat.compare_train()
            max_x_out = 0
            max_y_out = 0
            # print(t)
            t += 1
            in_out_torch = []
            test_torch = []
            test_nb = 0
            for test in task['test']:
                test_in_mat = test['input']
                test_x_in = len(test_in_mat[0])
                test_y_in = len(test_in_mat)
                test_id = self.task_id.replace('.json', f'_{test_nb}')
                test_nb += 1
                test_in_mat_torch = torch.LongTensor(test_in_mat).view(1, test_y_in, test_x_in).float()
                mat_decomp = MatrixDecomposition(test_in_mat_torch.unsqueeze(0))
                all_mat = mat_decomp.mat_tensor
                all_mat.insert(0, test_in_mat_torch)
                test_torch.append((all_mat, test_id))
            for train in task['train']:
                in_mat = train['input']
                out_mat = train['output']
                if len(train['input'][0]) == 1 or len(train['input']) == 1:
                    in_mat = np.kron(train['input'], [[1,1], [1,1]])
                x_in = len(in_mat[0])
                y_in = len(in_mat)
                x_out = len(out_mat[0])
                y_out = len(out_mat)
                in_mat_torch = torch.LongTensor(in_mat).view(1, y_in, x_in)
                mat_decomp = MatrixDecomposition(in_mat_torch.unsqueeze(0))
                train_all_mat = mat_decomp.mat_tensor
                train_all_mat.insert(0, in_mat_torch)
                out_mat_torch = torch.LongTensor(out_mat).view(y_out, x_out)
                in_out_torch.append((train_all_mat, out_mat_torch))
                if x_out > max_x_out:
                    max_x_out = x_out
                if y_out > max_y_out:
                    max_y_out = y_out
            self.dataset.append((test_torch, in_out_torch, (max_x_out, max_y_out), torch.tensor([*feat.good_params.values()]), feat.good_params))
        self.nb_of_tasks = len(self.dataset)        

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return self.nb_of_tasks
