import os

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as trasformers

class ResidualConvBlock(nn.Module):
    def __init__(self, in_c, out_c, is_res = True):
        super().__init__()

        self.same_channels = in_c == out_c
        self.is_res = is_res

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 1, 1),
            nn.BatchNorm2d(out_c),
            nn.GELU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_c, out_c, 3, 1, 1),
            nn.BatchNorm2d(out_c),
            nn.GELU()
        )

        self.re_conv = nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)

            if self.same_channels:
                x3 = x + x2
            else:
                x3 = self.re_conv(x) + x2
            return x3 / (2 ** 0.5)
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2
    
    def get_out_channels(self):
        return self.conv2[0].out_channels

class UnetUP(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        layers=[
            nn.ConvTranspose2d(in_c, out_c, 2, 2),
            ResidualConvBlock(out_c, out_c, False),
            ResidualConvBlock(out_c, out_c, False),
        ]

        self.seq = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = torch.cat((x, skip), dim=1)
        x = self.seq(x)
        return x
    
class UnetDown(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        layers = [
            ResidualConvBlock(in_c, out_c, False),
            ResidualConvBlock(out_c, out_c, False),
            nn.MaxPool2d(2)
        ]

        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)
    
class EmbedFC(nn.Module):
    def __init__(self, in_dim, emb_dim):
        super().__init__()

        layers = [
            nn.Linear(in_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]

        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)

class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        half= self.dim // 2
        # positional embedding 수식
        emb = math.log(10000) / (half - 1)
        emb= torch.exp(torch.arange(half, device=t.device) * -emb)
        emb= t[:, None] * emb[None, :]
        emb= torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    
class ContextUnet(nn.Module):
    def __init__(self, in_c, n_feat=256, n_cfeat=10, height=28):
        super().__init__()

        self.in_c = in_c
        self.n_feat = n_feat
        self.n_cfeat = n_cfeat
        self.height = height

        self.init_conv = ResidualConvBlock(in_c, n_feat, True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)

        self.to_vec = nn.Sequential(
            nn.AvgPool2d((4)),
            nn.GELU(), 
        )

        # learnable time embedding 
        # (하지만 SinusoidalPositionEmbedding을 주로 선호 함)
        self.time_embed1 = EmbedFC(1, 2*n_feat)
        self.time_embed2 = EmbedFC(1, 1*n_feat)
        # self.time_embed = SinusoidalPositionEmbedding(n_feat)
        self.context_embed1 = EmbedFC(n_cfeat, 2*n_feat)
        self.context_embed2 = EmbedFC(n_cfeat, 1*n_feat)

        # self.context_embed = SinusoidalPositionEmbedding(n_cfeat)

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(
                2 * n_feat, 2 * n_feat, 
                self.height // 4, self.height // 4
            ),
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU()
        )
        self.up2 = UnetUP(4 * n_feat, n_feat)
        self.up3 = UnetUP(2 * n_feat, n_feat)

        self.out = nn.Sequential(
            nn.Conv2d(2*n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_c, 3, 1, 1),
        )

    def forward(self, x, t, c=None):
        x = self.init_conv(x)
        down1 = self.down1(x)    
        down2 = self.down2(down1)  
        hiddenvec = self.to_vec(down2)

        if c is None:
            c = torch.zeros(x.shape[0], self.n_cfeat).to(x.device)
            
        cemb1 = self.context_embed1(c).view(-1, self.n_feat * 2, 1, 1) 
        temb1 = self.time_embed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.context_embed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.time_embed2(t).view(-1, self.n_feat, 1, 1)

        up1 = self.up1(hiddenvec)
        up2 = self.up2(cemb1 * up1 + temb1, down2) 
        up3 = self.up3(cemb2 * up2 + temb2, down1) 
        
        out = self.out(torch.cat((up3, x), 1))
        return out