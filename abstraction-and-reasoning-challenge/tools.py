#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 18:34:57 2020

@author: aktasos
"""

import torch
import numpy as np

x = [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
 [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
 [0, 0, 1, 0, 2, 2, 0, 1, 0, 0],
 [0, 0, 1, 0, 2, 2, 0, 1, 0, 0],
 [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
 [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

y = [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
 [0, 0, 0, 0, 2, 2, 0, 1, 0, 0],
 [0, 0, 0, 0, 2, 2, 0, 1, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
 [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]]

input1 = torch.tensor(x).view(1, 1, 10, 10)
input2 = torch.tensor(y).view(1, 1, 10, 10)

def plot_grid(mat, vmin=0, vmax=9):
    cmap = colors.ListedColormap(
        ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin, vmax)
    
    input_matrix = mat
    plt.imshow(input_matrix, cmap=cmap, norm=norm)
    plt.show()

def detect_borders(matrix):
    up, u_count =  matrix[0][0][0].unique(return_counts=True)
    right, r_count= matrix[0][0].transpose(0,1)[0].unique(return_counts=True)
    down, d_count = matrix[0][0][-1].unique(return_counts=True)
    left, l_count = matrix[0][0].transpose(0,1)[-1].unique(return_counts=True)
    values = torch.cat((up, right,down,left))
    unique_values = values.unique()
    count = torch.cat((u_count, r_count,d_count,l_count))
    border_dict = {}
    for uv in unique_values:
        border_dict[int(uv)] = 0 
        indices = np.where(values==int(uv))[0]
        for i in indices:
            border_dict[int(uv)] += int(count[i])
    max_color = max(border_dict, key=border_dict.get)
    if border_dict[max_color] < (np.ma.size(matrix)/2):
        return None, border_dict
    else:
        return max_color, border_dict       
    

def detect_grids(matrix):
    lines = []
    columns = []
    # temp_mat_lines = []
    # temp_mat_columns = []
    grid = False
    for index, line in enumerate(matrix[0][0]):
        unique_values, count = line.unique(return_counts=True)
        if len(unique_values)==1:
            lines.append((index, int(unique_values)))
        # else:
        #     temp_mat_lines.append(line)
            
    for index, column in enumerate(matrix[0][0].transpose(0,1)):
        if len(unique_values)==1:
            columns.append((index, int(unique_values)))  
         # else:
         #        temp_mat_columns.append(columns)
            
    
    # if lines:
    #     temp_mat_lines = np.array(temp_mat_lines).transpose(0,1)
    #     for index, line in enumerate(temp_mat):
    #         if len(unique_values)==1:
    #             columns.append(index)  
                            
    
    
    
    if lines and columns:
        grid = True
        
        
        return lines, columns, grid
    
    

 # elif max(count) >= (len(lines)/3)*2 and len(unique_values)<=3:
 #            lines.append(index)


max_c1, b1 = detect_borders(input1)
max_c2, b2 = detect_borders(input2)
lines1, columns1, grid1 = detect_grids(input1)
lines2, columns2, grid2 = detect_grids(input2)
