#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 14:29:19 2020

@author: aktasos
"""
import torch.nn as nn
# import torch.nn.functional as F


class FcNetwork(nn.Module):
    def __init__(self,in_feat=168, nb_of_fclayers=1000, out_feat=168):
        
        self.in_channels = in_feat
        self.hidden_layers_feat =672
        self.layer_output = 672
        self.output = out_feat
        self.nb_of_fclayers = nb_of_fclayers
        
        # super function. It inherits from nn.Module and we can access everythink in nn.Module
        super(FcNetwork,self).__init__()
        # Function.
    # determine number of convolution according to size of image
      
        
        self.fc = []
        self.fc += [nn.Linear(in_features = self.in_channels, out_features = self.hidden_layers_feat)]     
        
        for n in range(self.nb_of_fclayers-2) :
            
            self.fc += [nn.Linear(in_features = self.layer_output, out_features = self.layer_output)]     
        
         
        self.fc += [nn.Linear(in_features = self.layer_output, out_features = self.output)]  
          
        self.fc = nn.Sequential(*self.fc)



    def forward(self,x):
      
        x = self.fc(x)
        return x
    
