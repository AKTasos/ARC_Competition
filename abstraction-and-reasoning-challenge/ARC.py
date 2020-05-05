#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 19:04:55 2020

@author: aktasos
"""
import os
from pathlib import Path
from torch.utils.data import DataLoader
from descriptive_stats import correct_output
from multi_scale_networkconv import CnnFcNetwork
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tensorfromdata import TasksTensorDataset, FeaturesTensorDataset, data_openner
from plot_task import plot_pred
import numpy as np
from collections import OrderedDict


PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = Path(PATH)
TRAINING_PATH = DATA_PATH / 'training'
EVALUATION_PATH = DATA_PATH / 'evaluation'
TEST_PATH = DATA_PATH / 'test'
training_tasks = sorted(os.listdir(TRAINING_PATH))

FEATURES_DATA_PATH = "./training_results/params_data.json"

train_tasks = data_openner(training_tasks, TRAINING_PATH)
params_data = FeaturesTensorDataset(FEATURES_DATA_PATH, 0)
task_data = TasksTensorDataset(train_tasks[0])


#parameters = dictionary of parameters for DataLoader and optim (dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn, pin_memory, drop_last, timeout, worker_init_fn)
parameters = dict(
    lr=0.001,
    batch_size=1,
    shuffle=False,
    epochs=1000

    # ,nb_of_fclayers = [2,4]
    # ,act_fonction=["relu","glu","tanh","sigmoid","softmax"]
    # ,kernel_size = [4,8]
    )

data_loader = DataLoader(dataset=task_data, batch_size=parameters['batch_size'], shuffle=False)
in_data, out_data = next(iter(data_loader))

network = CnnFcNetwork(in_data)
optimizer = optim.Adam(network.parameters(), lr=parameters['lr'])

results=OrderedDict()

for epoch in range(parameters['epochs']):
    total_loss = 0
    total_correct = 0

    for batch in data_loader:
        in_data, out_data = batch
        
        #runs the batch in the CNN
        feats_in = network(in_data.float())
        feats_out = network(out_data.float())
        # plot_pred(preds.argmax(dim=3))
        
        #calculate Loss
        n = 0
        loss = 0
        
        loss = F.mse_loss(feats_in, feats_out)
        # loss.requires_grad = True
        optimizer.zero_grad()
        #BackProp
        loss.backward()
        #update weights
        optimizer.step()
        n += 1
       
        total_loss += loss.item()
        
        
        # for i in preds.view(-1):
        #     loss = F.cross_entropy(i, out_data.view(-1)[n])
        #     # loss.requires_grad = True
        #     optimizer.zero_grad()
        #     #BackProp
        #     loss.backward()
        #     #update weights
        #     optimizer.step()
        #     n += 1
        #     print(loss)
        #     total_loss += loss.item()
        
        
        total_correct += correct_output(feats_in, feats_out)

    results[epoch] = (feats_in, feats_out)
    print("epoch:", epoch, "/  total_correct:", total_correct, "/  Loss:", total_loss)
   