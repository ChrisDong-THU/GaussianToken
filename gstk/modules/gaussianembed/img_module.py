import torch
from torch import nn

import math


def swish(x):
    # swish
    return x * torch.sigmoid(x)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') == 0:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, dim, eps=1e-6)
        self.norm2 = nn.GroupNorm(32, dim, eps=1e-6)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=(3, 3), padding=1, bias=False)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=(3, 3), padding=1, bias=False)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = swish(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = swish(x)
        x = self.conv2(x)
        
        return residual + x


class Encoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv_in = nn.Conv2d(in_dim, hidden_dim, 3, 1, 1, bias=False)
        self.layers = nn.Sequential(
            ResBlock(hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 2, 1),
            ResBlock(hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 2, 1),
            ResBlock(hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 2, 1),
            ResBlock(hidden_dim),
        )
        self.norm_out = nn.GroupNorm(32, hidden_dim, eps=1e-6)
        self.conv_out = nn.Conv2d(hidden_dim, out_dim, 1)
        
        self.apply(weights_init)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.layers(x)
        x = self.norm_out(x)
        x = swish(x)
        x = self.conv_out(x)
        
        return x


class Decoder(nn.Module):
    def __init__(self, hidden_dim=64, out_dim=3, in_dim=16):
        super().__init__()
        layers = [
            ResBlock(hidden_dim),
            UpsamplerShuffle(2, hidden_dim),
            ResBlock(hidden_dim),
            UpsamplerShuffle(2, hidden_dim),
            ResBlock(hidden_dim),
            UpsamplerShuffle(2, hidden_dim),
            ResBlock(hidden_dim),
        ]
        self.layers = nn.Sequential(*layers)
        self.norm_out = nn.GroupNorm(32, hidden_dim, eps=1e-6)
        self.conv_in = nn.Conv2d(in_dim, hidden_dim, 3, 1, 1)
        self.conv_out = nn.Conv2d(hidden_dim, out_dim, 3, 1, 1)
        
        self.apply(weights_init)
        
    def forward(self, x):
        x = self.conv_in(x)
        x = self.layers(x)
        x = self.norm_out(x)
        x = swish(x)
        x = self.conv_out(x)
        
        return x


class UpsamplerShuffle(nn.Sequential):
    def __init__(self, scale, dim):

        m = []
        if (scale & (scale - 1)) == 0: # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(dim, 4 * dim, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        else:
            raise NotImplementedError

        super(UpsamplerShuffle, self).__init__(*m)
        

class UpsamplerDeConv(nn.Sequential):
    def __init__(self, scale, dim):

        m = []
        if (scale & (scale - 1)) == 0: # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.ConvTranspose2d(dim, dim, 4, 2, 1))
        else:
            raise NotImplementedError

        super(UpsamplerDeConv, self).__init__(*m)
        
        
class UpsamplerInterpolate(nn.Sequential):
    def __init__(self, scale, dim):

        m = []
        if (scale & (scale - 1)) == 0: # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Upsample(scale_factor=2.0, mode='nearest', align_corners=False))
                m.append(nn.Conv2d(dim, dim, 3, 1, 1))
        else:
            raise NotImplementedError

        super(UpsamplerInterpolate, self).__init__(*m)
        
        
def build_img_encoder(cfg):
    return Encoder(**cfg)

def build_img_decoder(cfg):
    return Decoder(**cfg)