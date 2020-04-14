#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 10:30:56 2020

@author: aktasos
"""

import matplotlib as mpl
from tensorfromdata import TensorDataset
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

data_path = "./training/params_data.json"

p = TensorDataset(data_path)

data=[]
for n in range(len(p.x)):
    train = []
    fig, ax = plt.subplots()
    for x in range(len(p.x[n])):
        
        train.append(((0, p.x[n][x]), (1, p.y[n][x])))
    lines = LineCollection(train, linewidths=(1))   
    ax.add_collection(lines)
    data.append(train)    
    plt.axis([0, 2, 0, 20])
    plt.show()    

# x = [int(x) for x in p.x[0]]
# y = [int(y) for y in p.y[0]]

# fig, ax = plt.subplots()
# lines = LineCollection(data, linewidths=(0.01))

# ax.add_collection(lines)

# plt.axis([0, 2, 0, 20])


# [p.x[0][0]]