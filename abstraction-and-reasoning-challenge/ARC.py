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
from tensorfromdata import TensorDataset
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict

from networkcfc import FcNetwork
import torch.optim as optim
import torch.nn.functional as F
import torch
import pandas as pd



def correct(preds, labels):
    c=torch.eq(preds,labels)
    return c.sum().item()

data_path = "./training/params_data.json"

params_data = TensorDataset(data_path)



#parameters = dictionary of parameters for DataLoader and optim (dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn, pin_memory, drop_last, timeout, worker_init_fn)
parameters = dict(
    lr = 0.00001
    ,batch_size = 10
    ,shuffle = False
    ,epochs = 10
    # ,nb_of_fclayers = [2,4]
    # ,act_fonction=["relu","glu","tanh","sigmoid","softmax"]
    # ,kernel_size = [4,8]
    )


  
data_loader = DataLoader(dataset=params_data, batch_size=parameters['batch_size'], shuffle=False)


fc = FcNetwork()

optimizer = optim.SGD(fc.parameters(), lr=parameters['lr'])


for epoch in range(parameters['epochs']):
     total_loss = 0
     total_correct = 0
  
     for batch in data_loader:
       
        in_data, out_data = batch
        
        #runs the batch in the CNN
        preds=fc(in_data.float())
        
        
        #calculate Loss
        loss = F.multilabel_soft_margin_loss(preds,out_data)
        print(loss)
        optimizer.zero_grad()
        
        #BackProp
        loss.backward()
        
        #update weights
        optimizer.step()
        
        total_loss += loss.item() * parameters['batch_size']
        total_correct += correct(preds, out_data)
        
        
        
     print("epoch:", epoch ,"/  total_correct:", total_correct, "/  Loss:", loss)
   






