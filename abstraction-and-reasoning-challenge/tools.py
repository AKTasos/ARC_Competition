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


x = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
      [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
      [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
      [0, 0, 1, 0, 2, 2, 0, 1, 0, 0],
      [0, 0, 1, 0, 2, 2, 0, 1, 0, 0],
      [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
      [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]

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

z = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 1, 1, 1, 0, 1, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 2, 2, 0, 1, 0, 0],
      [0, 0, 0, 0, 2, 2, 0, 1, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
      [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

input1 = torch.tensor(x).view(1, 1, 10, 10)
input2 = torch.tensor(y).view(1, 1, 10, 10)
input3 = torch.tensor(z).view(1, 1, 10, 10)

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
        return color, border_dict

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
        bg, border_dict = detect_borders(matrix)
    return bg

def trim_borders(matrix, bg, lines, columns):
    nb_of_lines, nb_of_columns = matrix.size()
    trimed_mat = matrix
    trim = []
    lines_trim = []
    trimed_lines = np.empty(0)
    trimed_columns = np.empty(0)
    columns_trim = []
    n = 0
    if lines:
        if lines[0][0] == 0 and lines[0][1] == bg:
            while lines[n][1] == bg and lines[n][0] == n:
                trim.append(n)
                lines_trim.append(n)
                n += 1
                if n == len(lines):
                    break
        if lines[-1][0] == nb_of_lines-1 and lines[-1][1] == bg:
            m = len(lines)-1
            last_line = nb_of_lines-1
            while lines[m][1] == bg and lines[m][0] == last_line:
                trim.append(last_line)
                lines_trim.append(m)
                last_line -= 1
                m -= 1  
        trimed_mat = np.delete(trimed_mat, trim, axis=0)
        trimed_lines = np.delete(lines, lines_trim, axis=0)
        for line in trimed_lines:
            line[0] -= n
            
        trim = []
        n = 0
    if columns:    
        if columns[0][0] == 0 and columns[0][1] == bg:
            while columns[n][1] == bg and columns[n][0] == n:
                trim.append(n)
                columns_trim.append(n)
                n += 1
                if n == len(columns):
                    break
        if columns[-1][0] == nb_of_columns-1 and columns[-1][1] == bg:
            m = len(columns)-1
            last_columns = nb_of_columns-1
            while columns[m][1] == bg and columns[m][0] == last_columns:
                trim.append(last_columns)
                columns_trim.append(m)
                last_columns -= 1
                m -= 1  
        trimed_mat = np.delete(trimed_mat.transpose(0, 1), trim, axis=0).transpose(0, 1)
        trimed_columns = np.delete(columns, columns_trim, axis=0)
        for column in trimed_columns:
            column[0] -= n
    
    if trimed_mat.size() == matrix.size():
        trimed_mat, trimed_lines, trimed_columns = matrix, np.array(lines), np.array(columns)
        
    return trimed_mat, trimed_lines, trimed_columns

def sub_matrices(matrix, lines, columns):
    try:
        line_split = [l[0] for l in lines]
    except:
        line_split =[lines[0]]
    try:
        column_split = [c[0] for c in columns]
    except:
        column_split = [columns[0]]
    mat = np.split(matrix, line_split, axis=0)
    # print(mat)
    sub_m = []
    for sub_mat in mat:
        for m in np.split(sub_mat, column_split, axis=1):
            if np.count_nonzero(m) != 0:
                sub_m.append(m)
    return sub_m

def plot_grid(mat, vmin=0, vmax=9):
    cmap = colors.ListedColormap(
        ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin, vmax)
    input_matrix = mat
    plt.imshow(input_matrix, cmap=cmap, norm=norm)
    plt.show()
        

class MatrixDecomposition():
    def __init__(self, matrix):
        self.matrix = matrix[0][0]
        self.color, self.border_dict = detect_borders(self.matrix)
        self.bg = background(self.matrix)
        self.lines, self.columns, self.grid = detect_grids(self.matrix)
        self.all_mat = []
        self.mat_tensor = []
       
        if self.lines or self.columns:
            self.trimed_mat, self.trimed_lines, self.trimed_columns = trim_borders(self.matrix, self.bg, self.lines, self.columns)
            self.all_mat.append(self.trimed_mat)
            if self.trimed_lines.size > 0 or self.trimed_columns.size > 0:
                self.sub_m = sub_matrices(self.trimed_mat, self.trimed_lines, self.trimed_columns)
                
                for mat in self.sub_m:
                    self.sub_lines, self.sub_columns, self.sub_grid = detect_grids(mat)
                    self.sub_trimed_mat, self.sub_trimed_lines, self.sub_trimed_columns = trim_borders(mat, self.bg, self.sub_lines, self.sub_columns,)
                    self.all_mat.append(self.sub_trimed_mat)
                    if self.sub_trimed_lines.size > 0 or self.sub_trimed_columns.size > 0:
                        self.sub_sub_m = sub_matrices(self.trimed_mat, self.trimed_lines, self.trimed_columns)
                        for smat in self.sub_sub_m: 
                            self.ssub_lines, self.ssub_columns, self.ssub_grid = detect_grids(smat)
                            self.ssub_trimed_mat, self.ssub_trimed_lines, self.ssub_trimed_columns = trim_borders(smat, self.bg, self.ssub_lines, self.ssub_columns)
                            self.all_mat.append(self.ssub_trimed_mat)

        for idx, m in enumerate(self.all_mat):
            self.all_mat[idx]=m.tolist()
        try:
            self.unique_mat = np.unique(self.all_mat, axis=0)
        except:
            self.unique_mat = np.unique(self.all_mat)
        self.unique_mat_clean = []
        for idx, l in enumerate(self.unique_mat):
            if len(l)>1:
                self.unique_mat_clean.append(l)
                l_tensor = torch.LongTensor(l).unsqueeze(0)
                if l_tensor.dim()<3:
                    l_tensor = l_tensor.unsqueeze(0)
                self.mat_tensor.append(l_tensor)
        


a = MatrixDecomposition(input1)       
# b = MatrixDecomposition(input2)
# c = MatrixDecomposition(input3)