#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 14:29:19 2020

@author: aktasos
"""
import os
from pathlib import Path
from torch.utils.data import DataLoader
from descriptive_stats import correct
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tensorfromdata import AllTasksGroupedWithTest, data_openner
from plot_task import plot_pred
import torch
import math
import numpy as np
from torch.autograd import Variable


PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = Path(PATH)
TRAINING_PATH = DATA_PATH / 'training'
EVALUATION_PATH = DATA_PATH / 'evaluation'
TEST_PATH = DATA_PATH / 'test'
training_tasks = sorted(os.listdir(TRAINING_PATH))

FEATURES_DATA_PATH = "./training_results/params_data.json"

train_tasks = data_openner(training_tasks, TRAINING_PATH)




class CnnFcNetwork(nn.Module):
    def __init__(self, in_channels=1 , kernel_size=1, nb_of_fclayers=2):

        self.in_channels = in_channels
        self.new_feat_map = self.in_channels*2
        self.kernel_size = kernel_size#[x+1 for x in range(3)]
        self.n_delta = None
        self.out_features = None
        self.nb_of_fclayers = nb_of_fclayers

        # super function. It inherits from nn.Module and we can access everythink in nn.Module
        super(CnnFcNetwork, self).__init__()
        # Function.
    # determine number of convolution according to size of image  int(math.log(self.img_size)/math.log(2)) 
        self.nb_of_conv = 10

    # creating layers

        self.cnnk1 = []
        for n in range(self.nb_of_conv):
            self.cnnk1 += [nn.Conv2d(self.in_channels, self.new_feat_map, kernel_size=1),
                nn.BatchNorm2d(self.new_feat_map),
                nn.LeakyReLU()]

            self.in_channels = self.new_feat_map
            self.new_feat_map = self.new_feat_map*2
            # self.layer_output = (self.layer_output-self.kernel_size)+1

        self.cnnk1 = nn.Sequential(*self.cnnk1)

        # self.layer_output = int(self.layer_output ** 2 * (self.new_feat_map/2))
          
        self.AdaptiveAvgPool1d = nn.AdaptiveAvgPool1d((1000))

        # self.fc = []
        # for n in range(self.nb_of_fclayers):
            
        #     self.fc += [nn.Linear(in_features=1000, out_features=1000)
        #                 , nn.LeakyReLU()]
           
        # self.fc = nn.Sequential(*self.fc)


    def forward(self, in_out):
        
        train_res = torch.empty((len(in_out),1000))
        x_res = torch.empty((len(in_out),1000))
        y_res = torch.empty((len(in_out),1000))
        n = 0
     
        
        for train in in_out:
            x, y = train
                
            x = self.cnnk1(x)
            x = x.view(1, 1, -1)
            x = self.AdaptiveAvgPool1d(x)
            
            y = self.cnnk1(y)
            y = y.view(1, 1, -1)
            y = self.AdaptiveAvgPool1d(y)
      
            train_res[n]=abs(x-y)
            x_res[n]=x
            y_res[n]=y
            n += 1
            
        return train_res, x_res, y_res
    
    



class OutputNetwork(nn.Module):
    def __init__(self, feats, in_channels=1 , kernel_size=1, nb_of_fclayers=10, out_size=(30,30)):

        self.in_channels = in_channels
        self.new_feat_map = self.in_channels*2
        self.kernel_size = kernel_size#[x+1 for x in range(3)]
        self.n_delta = None
        self.out_features = None
        self.nb_of_fclayers = nb_of_fclayers
        self.out_size = out_size
        self.out_x, self.out_y = out_size
        self.feats = feats
        self.nb_of_feats = len(feats[0])
        self.fc_in_size = (self.out_x * self.out_y).item()
        self.fc_out_size = (self.out_x*self.out_y*10).item()
        
        
        # super function. It inherits from nn.Module and we can access everythink in nn.Module
        super(OutputNetwork, self).__init__()
        # Function.
    # determine number of convolution according to size of image  int(math.log(self.img_size)/math.log(2)) 
        self.nb_of_conv = 10

    # creating layers

        self.cnnk1 = []
        for n in range(self.nb_of_conv):
            self.cnnk1 += [nn.Conv2d(self.in_channels, self.new_feat_map, kernel_size=1),
                nn.BatchNorm2d(self.new_feat_map),
                nn.LeakyReLU()]

            self.in_channels = self.new_feat_map
            self.new_feat_map = self.new_feat_map*2
            # self.layer_output = (self.layer_output-self.kernel_size)+1

        self.cnnk1 = nn.Sequential(*self.cnnk1)

        # self.layer_output = int(self.layer_output ** 2 * (self.new_feat_map/2))
          
        self.AdaptiveAvgPool2d_1 = nn.AdaptiveAvgPool2d((self.out_x, self.out_y))
        
        self.AdaptiveAvgPool1d_1 = nn.AdaptiveAvgPool1d(self.fc_in_size)
        
        self.fc = []
        self.fc += [nn.Linear(in_features=(self.fc_in_size + self.nb_of_feats), out_features=self.fc_in_size)
                        , nn.LeakyReLU()]
                              
        for n in range(self.nb_of_fclayers):
            
            self.fc += [nn.Linear(in_features=self.fc_in_size, out_features=self.fc_in_size)
                        , nn.LeakyReLU()]
        self.fc += [nn.Linear(in_features=self.fc_in_size, out_features=self.fc_out_size)
                        , nn.LeakyReLU()]   
        self.fc = nn.Sequential(*self.fc)
        
        self.featfc = []
        for n in range(self.nb_of_fclayers):
            
            self.featfc += [nn.Linear(in_features=(self.nb_of_feats), out_features=(self.nb_of_feats))
                        , nn.LeakyReLU()]
        self.featfc = nn.Sequential(*self.featfc)
        
        # self.AdaptiveAvgPool2d_2 = nn.AdaptiveAvgPool2d(((self.out_x*10),(self.out_y*10)))

    def forward(self, in_mat, out_size, feats, y=-1):
        out_x, out_y = out_size
        n=0
        for x in in_mat:
            
            x = x.float()
            x = self.cnnk1(x)
            x = self.AdaptiveAvgPool2d_1(x)        
            x = x.view(1,1,-1)
            if n==0:
                all_x = x.view(-1)
            else:
                all_x = torch.cat((all_x, x.view(-1)), dim=0)
        x = all_x.view(1,1,-1)
        x = self.AdaptiveAvgPool1d_1(x)                       
        f = self.featfc(feats)
        x = torch.cat((x.view(-1),f.view(-1)), dim=0)
        x = self.fc(x)

        if type(y) is torch.Tensor:
            out_y, out_x = y[0].shape
            x = F.adaptive_avg_pool1d(x.view(1, 1, -1),(out_x*out_y*10))
            
            
        # x = self.AdaptiveAvgPool2d_2(x)
        # x = F.adaptive_avg_pool1d(x, (out_x*out_y*10).item())
        x = x.view(1,out_x, out_y, 10)
   
        
        return x
    