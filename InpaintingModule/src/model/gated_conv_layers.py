from src.model.conv_layers import *
from torch.nn.utils import spectral_norm
import torch.nn as nn

class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, pad_type='replicate', activation='elu', norm='none', sc=False, sn=False):
        super(GatedConv2d, self).__init__()
        self.pad = None
        self.norm = None
        self.activation = None
        
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding) # pad : ['Replicate', 'Reflect', 'None']
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        elif activation == 'none':
            self.pad = None
            
        if norm == 'batch':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'none':
            self.norm = None
            
        if activation == 'elu':
            self.activation = nn.ELU(alpha=1.0, inplace=True) # activation : ['elu', 'relu', 'lrelu']
        elif activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'none':
            self.activation = None

        if sn:
            if sc:
                self.conv2d = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation))
                self.mask_conv2d = spectral_norm(Conv_SC(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation))
            else:
                self.conv2d = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation))
                self.mask_conv2d = spectral_norm(Conv_DS(in_channels, out_channels, kernel_size, stride, padding = 0,dilation=dilation))
        else:
            if sc:
                self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation)
                self.mask_conv2d = Conv_SC(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation)
            else:
                self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation)
                self.mask_conv2d = Conv_DS(in_channels, out_channels, kernel_size, stride, padding = 0,dilation=dilation)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x_in): # sigmoid(mask) * activation(conv)
        x = self.pad(x_in)
        conv = self.conv2d(x)
        mask = self.mask_conv2d(x)
        gated_mask = self.sigmoid(mask)
        x = gated_mask * conv
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class TransposeGatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, pad_type='replicate', activation='elu', norm='none', sc=False, sn=False, scale_factor=2):
        super(TransposeGatedConv2d, self).__init__()
        # Initialize the conv scheme
        self.scale_factor = scale_factor
        self.gated_conv2d = GatedConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, sc)

    def forward(self, x):
        x = F.interpolate(x, scale_factor = self.scale_factor, mode = 'nearest')
        x = self.gated_conv2d(x)
        return x