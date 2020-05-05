#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 14:29:19 2020

@author: aktasos
"""
import torch.nn as nn
import torch.nn.functional as F
import math

class CnnFcNetwork(nn.Module):
    def __init__(self, torch_input, kernel_size=1, in_feat=3*3, out_feat=9*9*2, nb_of_fclayers=5):

        self.in_channels = torch_input.shape[1]
        self.img_size = torch_input.shape[2]*torch_input.shape[3]
        self.new_feat_map = self.in_channels*2
        self.kernel_size = kernel_size#[x+1 for x in range(3)]
        self.layer_output = torch_input.shape[2]
        self.n_delta = None
        self.out_features = None
        self.output = out_feat
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
            self.layer_output = (self.layer_output-self.kernel_size)+1

        self.cnnk1 = nn.Sequential(*self.cnnk1)

        self.layer_output = int(self.layer_output ** 2 * (self.new_feat_map/2))
          
        self.AdaptiveAvgPool1d = nn.AdaptiveAvgPool1d((1000))

        self.fc = []
        for n in range(self.nb_of_fclayers):
            self.fc += [nn.Linear(in_features=1000, out_features=1000)
                        , nn.LeakyReLU()]
            
        self.fc += [nn.Linear(in_features=1000, out_features=30)
                        , nn.LeakyReLU()]
           
        self.fc = nn.Sequential(*self.fc)


    def forward(self, x):
        x = self.cnnk1(x)
        x = x.view(1, 1, -1)
        x = self.AdaptiveAvgPool1d(x)
        x = self.fc(x)
        
        return (x-min(x[0][0]))/(max(x[0][0])-min(x[0][0]))
    # .argmax(dim=3)
