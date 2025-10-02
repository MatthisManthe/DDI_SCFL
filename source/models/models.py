#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 15:56:46 2022

@author: rouge
Re-implementation of U-Net
"""


import torch
from torch import nn
import numpy as np

from monai.networks.layers.factories import Conv
from monai.networks.blocks import Convolution

from monai.networks.nets import AutoEncoder
import warnings
from typing import Optional, Sequence, Tuple, Union
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch import sigmoid
from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.simplelayers import SkipConnection
from monai.utils import alias, deprecated_arg, export

import monai.networks.blocks.dynunet_block
from monai.networks.blocks.dynunet_block import get_padding, get_output_padding


class CNN_TripleMNIST_Classification(nn.Module):
    def __init__(self, dropout=0):
        super(CNN_TripleMNIST_Classification, self).__init__()        
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=8,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),   
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(8, 16, 5, 1, 2),     
            nn.ReLU(),    
            nn.Dropout2d(p=dropout),                   
            nn.MaxPool2d(2),                
        )    
        self.conv3 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),
            nn.Dropout2d(p=dropout),                    
            nn.MaxPool2d(2),                
        )  
        self.linear = nn.Linear(32 * 8 * 12, 32)   
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.out = nn.Linear(32, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)  
        x = self.dropout(self.relu(self.linear(x)))
        output = self.out(x)
        return output    # return x for visualization



class CNN_GTA_Cityscapes_source_Classification(nn.Module):
    def __init__(self, sources=2, dropout=0):
        super(CNN_GTA_Cityscapes_source_Classification, self).__init__()        
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=3,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2                
            ),                              
            nn.LeakyReLU(),
            #nn.BatchNorm2d(16),
            nn.InstanceNorm2d(16)
            #nn.GroupNorm(1, 16)
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(
                in_channels=16,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2                
            ),                              
            nn.LeakyReLU(),
            #nn.BatchNorm2d(16),
            nn.InstanceNorm2d(16)
            #nn.GroupNorm(1, 16)
        )
        self.conv3 = nn.Sequential(         
            nn.Conv2d(16, 16, 5, 1, 2),     
            nn.LeakyReLU(),  
            nn.MaxPool2d(4),
            #nn.BatchNorm2d(16)
            nn.InstanceNorm2d(16),
            #nn.GroupNorm(1, 16),                   
        )    
        self.conv4 = nn.Sequential(         
            nn.Conv2d(16, 16, 5, 1, 2),     
            nn.LeakyReLU(),
            nn.MaxPool2d(4),
            #nn.BatchNorm2d(16),
            nn.InstanceNorm2d(16),
            #nn.GroupNorm(1, 16),                    
        ) 
        self.conv5 = nn.Sequential(         
            nn.Conv2d(16, 16, 5, 1, 2),     
            nn.LeakyReLU(),   
            nn.MaxPool2d(4),
            #nn.BatchNorm2d(16),
            nn.InstanceNorm2d(16),
            #nn.GroupNorm(1, 16),                         
        ) 
        self.linear = nn.Linear(16 * 8 * 16, 32)   
        self.relu = nn.LeakyReLU()
        self.out = nn.Linear(32, sources)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        #print(x.shape)
        x = x.view(x.size(0), -1)  
        x = self.relu(self.linear(x))
        output = self.out(x)
        return output    # return x for visualization


class FCN_Triple_MNIST_Segmentation(nn.Module):
    def __init__(self, nb_classes=11, norm="instance"):
        super(FCN_Triple_MNIST_Segmentation, self).__init__()    
        if norm == "instance":
            norm_layer = nn.InstanceNorm2d
        else:
            norm_layer = nn.BatchNorm2d
        
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2                
            ),                              
            nn.LeakyReLU(),
            norm_layer(16)
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(
                in_channels=16,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2                
            ),                              
            nn.LeakyReLU(),
            norm_layer(16)
        )
        self.conv3 = nn.Sequential(         
            nn.Conv2d(16, 32, 3, 1, 1),     
            nn.LeakyReLU(),  
            nn.MaxPool2d(2),
            norm_layer(32)                 
        )    
        self.conv4 = nn.Sequential(         
            nn.Conv2d(32, 32, 3, 1, 1),     
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            norm_layer(32)                    
        ) 
        self.conv5 = nn.Sequential(         
            nn.Conv2d(32, 32, 3, 1, 1),     
            nn.LeakyReLU(),   
            nn.MaxPool2d(2),
            norm_layer(32)                       
        ) 
        self.conv_trans31 = nn.Sequential(         
            nn.ConvTranspose2d(32, 32, 2, 2),     
            nn.LeakyReLU(),   
            norm_layer(32)                       
        ) 
        self.conv_trans41 = nn.Sequential(         
            nn.ConvTranspose2d(32, 32, 2, 2),     
            nn.LeakyReLU(),   
            norm_layer(32)                       
        ) 
        self.conv_trans42 = nn.Sequential(         
            nn.ConvTranspose2d(32, 32, 2, 2),     
            nn.LeakyReLU(),   
            norm_layer(32)                        
        ) 
        self.conv_trans51 = nn.Sequential(         
            nn.ConvTranspose2d(32, 32, 2, 2),     
            nn.LeakyReLU(),   
            norm_layer(32)                       
        ) 
        self.conv_trans52 = nn.Sequential(         
            nn.ConvTranspose2d(32, 32, 2, 2),     
            nn.LeakyReLU(),   
            norm_layer(32)                        
        ) 
        self.conv_trans53 = nn.Sequential(         
            nn.ConvTranspose2d(32, 32, 2, 2),     
            nn.LeakyReLU(),   
            norm_layer(32)                    
        ) 
        self.conv_agg = nn.Sequential(         
            nn.Conv2d(
                in_channels=112,              
                out_channels=16,            
                kernel_size=3,              
                stride=1,                   
                padding=1                
            ),                              
            nn.LeakyReLU(),
            norm_layer(16)
        )
        self.out = nn.Conv2d(
            in_channels=16,              
            out_channels=nb_classes,            
            kernel_size=1,              
            stride=1              
        ) 

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x_3 = self.conv3(x)
        x_4 = self.conv4(x_3)
        x_5 = self.conv5(x_4)
        
        x_51 = self.conv_trans51(x_5)
        x_52 = self.conv_trans52(x_51)
        x_53 = self.conv_trans53(x_52)
        
        x_41 = self.conv_trans41(x_4)
        x_42 = self.conv_trans42(x_41)
        
        x_31 = self.conv_trans31(x_3)

        x_agg = self.conv_agg(torch.cat([x, x_31, x_42, x_53], dim=1))
        
        output = self.out(x_agg)
        return output


class CNN_Triple_MNIST_source_Classification(nn.Module):
    def __init__(self, sources=2, dropout=0):
        super(CNN_Triple_MNIST_source_Classification, self).__init__()        
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2                
            ),                              
            nn.LeakyReLU(),
            nn.InstanceNorm2d(16)
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(
                in_channels=16,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2                
            ),                              
            nn.LeakyReLU(),
            nn.InstanceNorm2d(16)
        )
        self.conv3 = nn.Sequential(         
            nn.Conv2d(16, 16, 5, 1, 2),     
            nn.LeakyReLU(),  
            nn.MaxPool2d(4),
            nn.InstanceNorm2d(16)                
        )    
        self.conv4 = nn.Sequential(         
            nn.Conv2d(16, 16, 5, 1, 2),     
            nn.LeakyReLU(),
            nn.MaxPool2d(4),
            nn.InstanceNorm2d(16)               
        ) 
        self.linear = nn.Linear(16 * 4 * 6, 32)   
        self.relu = nn.LeakyReLU()
        self.out = nn.Linear(32, sources)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)  
        x = self.relu(self.linear(x))
        output = self.out(x)
        return output    # return x for visualization
    
    
# Redefinition of UnetBasicBlock class from Monai to add bias to convolutions.
def modified_get_conv_layer(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    kernel_size: Union[Sequence[int], int] = 3,
    stride: Union[Sequence[int], int] = 1,
    act: Optional[Union[Tuple, str]] = Act.PRELU,
    norm: Optional[Union[Tuple, str]] = Norm.INSTANCE,
    dropout: Optional[Union[Tuple, str, float]] = None,
    bias: bool = True, # Adding bias in every convolutional layer ...
    conv_only: bool = True,
    is_transposed: bool = False,
):
    padding = get_padding(kernel_size, stride)
    output_padding = None
    if is_transposed:
        output_padding = get_output_padding(kernel_size, stride, padding)
    return Convolution(
        spatial_dims,
        in_channels,
        out_channels,
        strides=stride,
        kernel_size=kernel_size,
        act=act,
        norm=norm,
        dropout=dropout,
        bias=bias,
        conv_only=conv_only,
        is_transposed=is_transposed,
        padding=padding,
        output_padding=output_padding,
    )

# %% Layers

class DoubleConv(nn.Sequential):
    def __init__(self, dim, in_features, out_features, strides, kernel_size, norm):
        
        super().__init__()
        
        conv1 = Convolution(
            spatial_dims=dim,
            in_channels=in_features,
            out_channels=out_features,
            strides=strides,
            kernel_size=kernel_size,
            adn_ordering="NDA",
            act=("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
            norm=(norm, {"affine": True}),
            
        )
        
        conv2 = Convolution(
            spatial_dims=dim,
            in_channels=out_features,
            out_channels=out_features,
            strides=1,
            kernel_size=kernel_size,
            adn_ordering="NDA",
            act=("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
            norm=(norm, {"affine": True}),
        )
        
        self.add_module('conv1', conv1)
        self.add_module('conv2', conv2)
        
        
class ConvUp(nn.Module):
    def __init__(self, dim, in_features, out_features, strides, kernel_size, conv_type, norm):
        
        super().__init__()
        
        self.conv_type = conv_type
        
        self.upsample = nn.Upsample(scale_factor=strides)
        
        self.conv1 = Convolution(
            spatial_dims=dim,
            in_channels=in_features,
            out_channels=out_features,
            strides=1,
            kernel_size=kernel_size,
            adn_ordering="NDA",
            act=("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
            norm=(norm, {"affine": True}),
            
        )
        
        if conv_type == "double":
            self.conv2 = Convolution(
                spatial_dims=dim,
                in_channels=out_features,
                out_channels=out_features,
                strides=1,
                kernel_size=kernel_size,
                adn_ordering="NDA",
                act=("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
                norm=(norm, {"affine": True}),
            )
        
    def forward(self, x):
        
        x = self.upsample(x)
        x = self.conv1(x)
        if self.conv_type == "double":
            x = self.conv2(x)

        return x
    
    
class ConvUp_final(nn.Module):
    def __init__(self, dim, in_features, out_features, img_size, kernel_size, conv_type, norm):
        
        super().__init__()
        
        self.conv_type = conv_type
        
        self.upsample = nn.Upsample(size=img_size)
        
        self.conv1 = Convolution(
            spatial_dims=dim,
            in_channels=in_features,
            out_channels=out_features,
            strides=1,
            kernel_size=kernel_size,
            adn_ordering="NDA",
            act=("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
            norm=(norm, {"affine": True}),
            
        )
        
        if conv_type == "double":
            self.conv2 = Convolution(
                spatial_dims=dim,
                in_channels=out_features,
                out_channels=out_features,
                strides=1,
                kernel_size=kernel_size,
                adn_ordering="NDA",
                act=("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
                norm=(norm, {"affine": True}),
            )
        
    def forward(self, x):
        
        x = self.upsample(x)
        x = self.conv1(x)
        if self.conv_type == "double":
            x = self.conv2(x)

        return x


# %% Modules

# Base Modules
class Encoder(nn.Module):
    def __init__(self, dim, in_channel, img_size, features, strides, kernel_size, depth, bottleneck_size, conv_type, norm):
        super(Encoder, self).__init__()
        
        self.dim = dim
        
        if depth + 1 != len(features):
            raise ValueError("Size of depth + 1 and features  must be equal")
            
        self.conv_input = Conv["conv", dim](in_channel, features[0], kernel_size=1)

        if conv_type == "single":
            self.blocks = nn.ModuleList(
                [
                    Convolution(
                       spatial_dims=dim,
                       in_channels=features[i],
                       out_channels=features[i+1],
                       strides=strides[i],
                       kernel_size=kernel_size[i],
                       adn_ordering="NDA",
                       act=("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
                       norm=(norm, {"affine": True}),
                   ) for i in range(depth)
                ])
        elif conv_type == "double":
            self.blocks = nn.ModuleList(
                [
                    DoubleConv(dim, features[i], features[i + 1], strides[i], kernel_size[i], norm)
                    for i in range(depth)
                ])
        
        self.flatten = nn.Flatten()
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.1)
        
        size = self.get_latent_space(img_size, features, strides, depth)
        self.linear = nn.Linear(size, bottleneck_size)

    def forward(self, x: torch.Tensor):
        
        x = self.leakyrelu(self.conv_input(x))
        
        for b in self.blocks:
            x = b(x)

        x = self.flatten(x)
        z = self.leakyrelu(self.linear(x))

        return z
    
    def get_latent_space(self, img_size, features, strides, depth):
        
        for s in strides:
            img_size = [img_size[i] // s for i in range(self.dim)]
                
        size = features[depth] * np.prod(img_size)
        return size
    
    
    
class VariationalEncoder(nn.Module):
    def __init__(self, dim, in_channel, img_size, features, strides, kernel_size, depth, bottleneck_size, conv_type, norm):
        super(VariationalEncoder, self).__init__()
        
        self.dim = dim
        
        if depth + 1 != len(features):
            raise ValueError("Size of depth + 1 and features must be equal")
            
        self.conv_input = self.conv_input = Conv["conv", dim](in_channel, features[0], kernel_size=1)

        if conv_type == "single":
            self.blocks = nn.ModuleList(
                [
                    Convolution(
                       spatial_dims=dim,
                       in_channels=features[i],
                       out_channels=features[i+1],
                       strides=strides[i],
                       kernel_size=kernel_size[i],
                       adn_ordering="NDA",
                       act=("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
                       norm=(norm, {"affine": True}),
                   ) for i in range(depth)
                ])
        elif conv_type == "double":
            self.blocks = nn.ModuleList(
                [
                    DoubleConv(dim, features[i], features[i + 1], strides[i], kernel_size[i], norm)
                    for i in range(depth)
                ])
        
        self.flatten = nn.Flatten()
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.1)
        
        size = self.get_latent_space(img_size, features, strides, depth)
        self.linear1 = nn.Linear(size, bottleneck_size)
        self.linear2 = nn.Linear(size, bottleneck_size)

    def forward(self, x: torch.Tensor):
        
        x = self.leakyrelu(self.conv_input(x))
        
        for b in self.blocks:
            x = b(x)
            
        x = self.flatten(x)
        mu = self.linear1(x)
        logvar = self.linear2(x)
        sigma = torch.exp(0.5 * logvar)
        
        eps = torch.randn_like(sigma)
        z = mu + (sigma * eps)

        return z, mu, logvar
    
    def get_latent_space(self, img_size, features, strides, depth):
        
       for s in strides:
           img_size = [img_size[i] // s for i in range(self.dim)]
               
       size = features[depth] * np.prod(img_size)
       return size
    
    
    
class Decoder(nn.Module):
    def __init__(self, dim, in_channel, img_size, features, strides, kernel_size, depth, bottleneck_size, conv_type, norm):
        super(Decoder, self).__init__()

        self.img_size = img_size
        self.features = features
        self.strides = strides
        self.depth = depth
        self.dim = dim
        
        shape = self.get_shape_latent_space(img_size, features, strides, depth)
        size = np.prod(shape)
        self.linear = nn.Linear(bottleneck_size, size)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.1)
        
        self.blocks = nn.ModuleList(
            [
                ConvUp(dim, features[depth - i], features[depth - i - 1], strides[depth - i - 1], kernel_size[depth - i - 1], conv_type, norm)
                for i in range(depth - 1)
            ])
        
        self.conv = ConvUp_final(dim, features[1], features[0], img_size, kernel_size[0], conv_type, norm)

        self.final_conv_1 = Conv["conv", dim](features[0], in_channel, kernel_size=1)

    def forward(self, z: torch.Tensor):
        
        z = self.leakyrelu(self.linear(z))
        batch, size = z.shape
        shape = [batch] + self.get_shape_latent_space(self.img_size, self.features, self.strides, self.depth)
        x = z.view(shape)
    
        for b in self.blocks:
            x = b(x)
            
        x = self.leakyrelu(self.conv(x))
            
        x_final = self.final_conv_1(x)

        return x_final
    
    def get_shape_latent_space(self, img_size, features, strides, depth):
        
        for s in strides:
            img_size = [img_size[i] // s for i in range(self.dim)]
        shape = [features[depth]] + img_size
        return shape    
    
    
# %% Models

class AutoEncoder(nn.Module):
    def __init__(self, dim, in_channel, img_size, features, strides, kernel_size, depth, bottleneck_size, conv_type, norm):
        super(AutoEncoder, self).__init__()
        
        self.encoder = Encoder(dim, in_channel, img_size, features, strides, kernel_size, depth, bottleneck_size, conv_type, norm)
        self.decoder = Decoder(dim, in_channel, img_size, features, strides, kernel_size, depth, bottleneck_size, conv_type, norm)
        
    def forward(self, x: torch.Tensor):
            
        z = self.encoder(x)
        x_final = self.decoder(z)
            
        return x_final, z


class VariationalAutoEncoder(nn.Module):
    def __init__(self, dim, in_channel, img_size, features, strides, kernel_size, depth, bottleneck_size, conv_type, norm):
        super(VariationalAutoEncoder, self).__init__()
        
        self.encoder = VariationalEncoder(dim, in_channel, img_size, features, strides, kernel_size, depth, bottleneck_size, conv_type, norm)
        self.decoder = Decoder(dim, in_channel, img_size, features, strides, kernel_size, depth, bottleneck_size, conv_type, norm)
        
    def forward(self, x: torch.Tensor):
            
        z, mu, logvar = self.encoder(x)
        if self.training:
            x_final = self.decoder(z)
        else:
            x_final = self.decoder(mu)
            
        return x_final, z, mu, logvar
