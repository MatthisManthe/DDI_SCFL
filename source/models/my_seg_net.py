import torch
from torch import nn

from monai.networks.layers.factories import Conv
from monai.networks.blocks import Convolution

from efficientvit.models.nn import MBConv
from efficientvit.models.nn.act import build_act
from efficientvit.models.nn.norm import build_norm


class DoubleConv(nn.Sequential):
    def __init__(self, dim, in_features, out_features, strides, kernel_size, dropout=0):
        
        super().__init__()
        
        conv1 = MBConv(
            in_channels=in_features,
            out_channels=out_features,
            stride=strides,
            kernel_size=kernel_size,
            norm="ln2d",
            act_func="relu",
            dropout=dropout
        )
        
        conv2 = MBConv(
            in_channels=out_features,
            out_channels=out_features,
            stride=1,
            kernel_size=kernel_size,
            norm="ln2d",
            act_func="relu",
            dropout=dropout
        )
        
        self.add_module('conv1', conv1)
        self.add_module('conv2', conv2)
        
        
class Conv_Up_with_skip(nn.Module):
    def __init__(self, dim, in_features, out_features, strides, kernel_size, dropout=0):
        
        super().__init__()
    
        self.conv_trans = torch.nn.ConvTranspose2d(
            in_channels=in_features,
            out_channels=out_features,
            stride=strides,
            kernel_size=strides,
            bias=False
        )
        
        self.norm = build_norm("ln2d", num_features=out_features)
        self.act = build_act("relu")
        
        self.conv = MBConv(
            in_channels=out_features * 2,
            out_channels=out_features,
            stride=1,
            kernel_size=kernel_size,
            norm="ln2d",
            act_func="relu",
            dropout=dropout
        )
        
    def forward(self, x, x_encoder):
        x_0 = self.conv_trans(x)
        x_0 = self.norm(x_0)
        x_0 = self.act(x_0)
        x_1 = self.conv(torch.cat([x_encoder, x_0], dim=1))

        return x_1
    
    
    
class Opt_UNet(nn.Module):
    def __init__(self, spatial_dims, in_channels, out_channels, features, strides, kernel_size, dropout):
        super(Opt_UNet, self).__init__()

        self.conv1 = DoubleConv(spatial_dims, in_channels, features[0], strides[0], kernel_size[0])
        self.conv2 = DoubleConv(spatial_dims, features[0], features[1], strides[1], kernel_size[1], dropout=0.1*dropout)
        self.conv3 = DoubleConv(spatial_dims, features[1], features[2], strides[2], kernel_size[2], dropout=0.1*dropout)
        self.conv4 = DoubleConv(spatial_dims, features[2], features[3], strides[3], kernel_size[3], dropout=0.1*dropout) 
        self.conv5 = DoubleConv(spatial_dims, features[3], features[4], strides[4], kernel_size[4], dropout=0.1*dropout)  
    
        self.up_1 = Conv_Up_with_skip(spatial_dims, features[4], features[3], strides[4], kernel_size[4], dropout=0.1*dropout)
        self.up_2 = Conv_Up_with_skip(spatial_dims, features[3], features[2], strides[3], kernel_size[3], dropout=dropout)
        self.up_3 = Conv_Up_with_skip(spatial_dims, features[2], features[1], strides[2], kernel_size[2], dropout=dropout)
        self.up_4 = Conv_Up_with_skip(spatial_dims, features[1], features[0], strides[1], kernel_size[1])
        
        self.final_conv_1 = Conv["conv", spatial_dims](features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        
        x6 = self.up_1(x5, x4)
        x7 = self.up_2(x6, x3)
        x8 = self.up_3(x7, x2)
        x9 = self.up_4(x8, x1)
        
        x_final = self.final_conv_1(x9)

        return x_final


"""
class BaseUnet(nn.Module):
     
    def __init__(self, dim, in_channel, features, strides, kernel_size):
        super(BaseUnet, self).__init__()
        
        self.shallowencoder = ShallowEncoder(dim, in_channel, features, strides, kernel_size)
        self.deepencoder = DeepEncoder(dim, in_channel, features, strides, kernel_size)
        self.decoder = Decoder_UNet(dim, in_channel, features, strides, kernel_size)
        
    def forward(self, x: torch.Tensor):
            
        x4, x3, x2, x1 = self.shallowencoder(x)
        x6, x5 = self.deepencoder(x4)
        x_final = self.decoder(x6, x1, x2, x3, x4, x5)
            
        return x_final
    
    def load_shallowencoder_weights(self, shallowencoder_weights):
        self.shallowencoder.load_state_dict(shallowencoder_weights)
        
    def load_deepencoder_weights(self, deepencoder_weights):
        self.deepencoder.load_state_dict(deepencoder_weights)
    
    def freeze_shallowencoder_weights(self):
        for p in self.shallowencoder.parameters():
            p.requires_grad = False
            
    def freeze_deepencoder_weights(self):
        for p in self.deepencoder.parameters():
            p.requires_grad = False
       
"""