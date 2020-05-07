#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 18:34:57 2020

@author: aktasos
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


x = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
     [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
     [0, 0, 1, 0, 2, 2, 0, 1, 0, 0],
     [0, 0, 1, 0, 2, 2, 0, 1, 0, 0],
     [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
     [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

y = [[1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
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
        ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin, vmax)
    input_matrix = mat
    plt.imshow(input_matrix, cmap=cmap, norm=norm)
    plt.show()

def detect_borders(image):
    matrix = image[0][0]
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
        color = "no clear borders"
    else:
        color = max_color
        return color, border_dict


def detect_grids(matrix):
    lines = []
    columns = []
    grid = False
    for index, line in enumerate(matrix[0][0]):
        unique_values = line.unique()
        if len(unique_values) == 1:
            lines.append((index, int(unique_values)))
    for index, column in enumerate(matrix[0][0].transpose(0, 1)):
        unique_values = column.unique()
        if len(unique_values) == 1:
            columns.append((index, int(unique_values)))
    if lines and columns:
        grid = True
    return lines, columns, grid

def background(matrix):
    if 0 in matrix.unique():
        bg = 0
    else:
        bg, border_dict = detect_borders(matrix)
    return bg

def trim_borders(matrix):
    bg = background(matrix)
    lines, columns, grid = detect_grids(matrix)
    nb_of_lines, nb_of_columns = matrix[0][0].size()
    trimed_mat = matrix[0][0]
    trim = []
    
    if lines[0][0] == 0 and lines[0][1] == bg:
        n = 0
        while lines[n][1] == bg and lines[n][0] == n:
            trim.append(n)
            n += 1
    if lines[-1][0] == nb_of_lines-1 and lines[-1][1] == bg:
        n = len(lines)-1
        last_line = nb_of_lines-1
        while lines[n][1] == bg and lines[n][0] == last_line:
            trim.append(last_line)
            last_line -= 1
            n -= 1  
    trimed_mat = np.delete(trimed_mat, trim, axis=0)
    
    trim = []
    if columns[0][0] == 0 and columns[0][1] == bg:
        n = 0
        while columns[n][1] == bg and columns[n][0] == n:
            trim.append(n)
            n += 1
    if columns[-1][0] == nb_of_columns-1 and columns[-1][1] == bg:
        n = len(columns)-1
        last_columns = nb_of_columns-1
        while columns[n][1] == bg and columns[n][0] == last_columns:
            trim.append(last_columns)
            last_columns -= 1
            n -= 1  
    trimed_mat = np.delete(trimed_mat.transpose(0, 1), trim, axis=0).transpose(0, 1)
    return trimed_mat

def sub_matrices(matrix):
    bg = background(matrix)
    lines, columns, grid = detect_grids(matrix)
    nb_of_lines, nb_of_columns = matrix[0][0].size()
    trimed_mat = matrix[0][0]
    cuts = []
    n = 0
    while lines[n][0] == n:
        del lines[0]
        n += 1
        cuts.append(n)
    
    if columns[-1][0] == nb_of_columns-1  
    
    if lines[0][0] == 0 and lines[0][1] == bg:
        n = 0
        while lines[n][1] == bg and lines[n][0] == n:
            trim.append(n)
            n += 1
    

border_col1, b1 = detect_borders(input1)
border_col2, b2 = detect_borders(input2)
lines1, columns1, grid1 = detect_grids(input1)
lines2, columns2, grid2 = detect_grids(input2)
a = trim_borders(input1)
b = trim_borders(input2) 

c = sub_matrices(input2)
d = sub_matrices(input2)
