#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 14:29:19 2020

@author: aktasos
"""
import torch.nn as nn
import torch.nn.functional as F
import torch

class OutputNetwork(nn.Module):
    def __init__(self, feats, in_channels=1, kernel_size=1, nb_of_fclayers=10, out_size=(30, 30)):

        self.in_channels = in_channels
        self.new_feat_map = self.in_channels*2
        self.kernel_size = kernel_size#[x+1 for x in range(3)]
        self.nb_of_fclayers = nb_of_fclayers
        self.out_x, self.out_y = out_size
        self.nb_of_feats = len(feats[0])
        self.fc_in_size = (self.out_x * self.out_y).item()
        self.fc_out_size = (self.out_x*self.out_y*10).item()
        self.nb_of_conv = 10
        # super function. It inherits from nn.Module and we can access everythink in nn.Module
        super(OutputNetwork, self).__init__()
        # Function.

    # creating layers

        self.cnnk1 = []
        for n in range(self.nb_of_conv):
            self.cnnk1 += [nn.Conv2d(self.in_channels, self.new_feat_map, self.kernel_size),
                           nn.BatchNorm2d(self.new_feat_map),
                           nn.LeakyReLU()]
            self.in_channels = self.new_feat_map
            self.new_feat_map = self.new_feat_map*2
        self.cnnk1 = nn.Sequential(*self.cnnk1)

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

    def forward(self, in_mat, out_size, feats, y=-1):
        out_x, out_y = out_size
        n = 0
        for x in in_mat:
            x = x.float()
            x = self.cnnk1(x)
            x = self.AdaptiveAvgPool2d_1(x)
            x = x.view(1, 1, -1)
            if n == 0:
                all_x = x.view(-1)
            else:
                all_x = torch.cat((all_x, x.view(-1)), dim=0)
        x = all_x.view(1, 1, -1)
        x = self.AdaptiveAvgPool1d_1(x)           
        f = self.featfc(feats)
        x = torch.cat((x.view(-1), f.view(-1)), dim=0)
        x = self.fc(x)
        if type(y) is torch.Tensor:
            out_y, out_x = y[0].shape
            x = F.adaptive_avg_pool1d(x.view(1, 1, -1), (out_x*out_y*10))
        x = x.view(1, out_x, out_y, 10)
        return x
    