import os
import torch
from torch import nn
import numpy as np


class ResidualBlock(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3,3), stride=(1,1), padding='same')
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(1,1), stride=(1,1))
        self.relu2 = nn.ReLU()

    def forward(self, x):
        y = self.relu1(self.conv1(x))
        return x + self.relu2(self.conv2(y))

class ConvBlock(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(3,3), stride=(1,1), padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(1,1), stride=(1,1)),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layers(x)

class Encoder(nn.Module):
    def __init__(self, in_channels=2, filters=[32, 64, 128, 256], kernel_size=(4,4), stride=(2,2), residual=False, *args, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.filters = [filters[0],] + filters
        self.init_conv = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.filters[0], kernel_size=1, stride=1, padding=0), nn.ReLU())
        self.convs = nn.ModuleList([nn.Sequential(
                            nn.Conv2d(in_channels=self.filters[k], out_channels=self.filters[k+1], kernel_size=kernel_size, stride=stride, padding=(1,1)), nn.ReLU()
        ) for k in range(len(filters))])
        self.blocks = nn.ModuleList([ResidualBlock(filters[k]) for k in range(len(filters))]) if residual \
                            else nn.ModuleList([ConvBlock(filters[k]) for k in range(len(filters))])

    def forward(self, x, return_intermediate=False):
        x = self.init_conv(x)
        if not return_intermediate:
            for k in range(len(self.convs)):
                x = self.convs[k](x)
                x = self.blocks[k](x)
            return x
        else:
            output = []
            for k in range(len(self.convs)):
                x = self.convs[k](x)
                x = self.blocks[k](x)
                output.append(x)
            return output

class Decoder(nn.Module):
    def __init__(self, out_channels=2, filters=[256, 128, 64, 32], kernel_size=(4,4), stride=(2,2), residual=False, *args, **kwargs):
        super().__init__()
        self.out_channels = out_channels
        self.filters = filters + [filters[-1],]
        self.convs = nn.ModuleList([nn.Sequential(
                            nn.ConvTranspose2d(in_channels=self.filters[k], out_channels=self.filters[k+1], kernel_size=kernel_size, stride=stride, padding=(1,1)), nn.ReLU()
        ) for k in range(len(filters))])
        self.outconv = nn.Sequential(nn.Conv2d(in_channels=filters[-1], out_channels=out_channels, kernel_size=1, stride=1, padding=0), nn.ReLU())
        self.blocks = nn.ModuleList([ResidualBlock(filters[k]) for k in range(len(filters))]) if residual \
                            else nn.ModuleList([ConvBlock(filters[k]) for k in range(len(filters))])

    def forward(self, x):
        for k in range(len(self.convs)):
            x = self.blocks[k](x)
            x = self.convs[k](x)
        return self.outconv(x)

class AutoEncoder(nn.Module):
    def __init__(self, in_channels=2, filters=[32,64,128,256], residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = Encoder(in_channels=in_channels, filters=filters, residual=residual)
        self.decoder = Decoder(out_channels=in_channels, filters=filters[::-1], residual=residual)

    def forward(self, x):
        out_enc = self.encoder(x)
        x_hat = self.decoder(out_enc)
        return x_hat

    def _get_intermediate_features(self, x):
        self.eval()
        features = self.encoder(x, return_intermediate=True)
        return features

    @staticmethod
    def _get_exp_number(path):
        n = 0
        for dirname in [_dir for _dir in os.listdir(path) if os.path.isdir(os.path.join(path,_dir))]:
            split = dirname.split('_')
            if len(split)==2 and split[0]=='exp' and split[1].isdigit() and int(split[1])>n:
                n = int(split[1])
        return str(n+1)

class RadImageNet(nn.Module):
    def __init__(self, path, in_channels, out_indices=[1, 2, 3, 4], out_class=True, fill='auto', *args, **kwargs):
        ''''
        Possible modes :
            - pad (with zeros)
            - repeat
            - repeat_0 or repeat_1 to repeat a particular channel
            - auto : repeat if n_channels = 1 and pad if n_channels=2 
        '''
        super().__init__(*args, **kwargs)
        self.resnet = torch.load(path).cuda()
        self.in_channels = in_channels
        self.out_indices = out_indices
        self.fill = fill
        self.outputs = {}
        for i in out_indices:
            getattr(self.resnet, 'layer'+str(i))[-1].relu.register_forward_hook(self._get_layer_output('layer_'+str(i)))
        if out_class:
            getattr(self.resnet, 'fc').register_forward_hook(self._get_layer_output('fc'))
        self._define_format_mode()

    def _get_layer_output(self, name):
        def hook(model, input, output):
            self.outputs[name] = output.detach()
        return hook

    def _get_intermediate_features(self, x):
        x = self.format_input(x)
        self.resnet(x)
        return [v.detach().cpu() for v in self.outputs.values()]

    def format_input(self, x):
        if self.mode==0:
            return torch.nn.functional.pad(x, (0,0,0,0,0,3-self.in_channels,0,0), mode='constant', value=0)
        elif self.mode==1:
            return x.repeat((1,np.ceil(3/self.in_channels).astype(int),1,1))[:,:3,:]

    def _define_format_mode(self):
        if self.fill=='auto':
            self.mode = 1 if self.in_channels==1 else 0
        elif self.fill=='pad':
            self.mode = 0
        elif self.fill=='repeat':
            self.mode = 1
        
    def forward(self, x):
        x = self.format_input(x)
        print(x.size())
        return self.resnet(x)

    @staticmethod
    def _get_exp_number(path):
        n = 0
        for dirname in [_dir for _dir in os.listdir(path) if os.path.isdir(os.path.join(path,_dir))]:
            split = dirname.split('_')
            if len(split)==2 and split[0]=='exp' and split[1].isdigit() and int(split[1])>n:
                n = int(split[1])
        return str(n+1)


