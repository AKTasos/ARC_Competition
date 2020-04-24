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
    def __init__(self, torch_input, kernel_size=1, in_feat=3*3, out_feat=9*9*2, nb_of_fclayers=10):

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
    # determine number of convolution according to size of image
        self.nb_of_conv = int(math.log(self.img_size)/math.log(2))

    # creating layers

        self.cnn = []
        for n in range(self.nb_of_conv):
            self.cnn += [nn.Conv2d(self.in_channels, self.new_feat_map, self.kernel_size),
                nn.BatchNorm2d(self.new_feat_map),
                nn.LeakyReLU()]

            self.in_channels = self.new_feat_map
            self.new_feat_map = self.new_feat_map*2
            self.layer_output = (self.layer_output-self.kernel_size)+1

        self.cnn = nn.Sequential(*self.cnn)

        self.layer_output = int(self.layer_output ** 2 * (self.new_feat_map/2))
        self.n_delta = self.layer_output // nb_of_fclayers
        self.out_features = self.layer_output - self.n_delta


        self.fc = []
        for n in range(self.nb_of_fclayers):
            if n == self.nb_of_fclayers-1:
                self.out_features = self.output
            self.fc += [nn.Linear(in_features = self.layer_output, out_features = self.out_features)
                        , nn.LeakyReLU()]
            self.layer_output -= self.n_delta
            self.out_features -= self.n_delta

        self.fc = nn.Sequential(*self.fc)


    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x.view(x.shape[0],9,9,2)
    # .argmax(dim=3)
