#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 14:29:19 2020

@author: aktasos
"""
import torch.nn as nn

import numpy as np# create dummy data for training
x_values = [i for i in range(11)]
x_train = np.array(x_values, dtype=np.float32)
x_train = x_train.reshape(-1, 1)

y_values = [2*i + 1 for i in x_values]
y_train = np.array(y_values, dtype=np.float32)
y_train = y_train.reshape(-1, 1)

class LinearRegression(nn.Module):
    """simple linear regression model using PyTorch"""
    def __init__(self, inputSize, outputSize):
        # super function. It inherits from nn.Module and we can access everythink in nn.Module
        super(LinearRegression, self).__init__()
        # Function.
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out
