"""Vendored from paper/pulse/sparc-nn/models/unet1d.py; MIT, Copyright (c) 2025 Han Bui.
See LICENSE-PULSE. Source implementation below is unchanged.
"""
import torch 
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(ConvBlock, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1),
            # nn.BatchNorm1d(mid_channels),
            nn.InstanceNorm1d(mid_channels),
            # nn.GroupNorm(num_groups=8, num_channels=mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm1d(out_channels),
            nn.InstanceNorm1d(out_channels),
            # nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(ResidualBlock, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm1d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm1d(out_channels)
        )
        
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
            
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + residual
        out = self.relu(out)
        return out


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            ResidualBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose1d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ResidualBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffN = x2.size()[2] - x1.size()[2]
        x1 = F.pad(x1, (diffN // 2, diffN - diffN // 2))
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet1D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.inc = ResidualBlock(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        self.bottleneck = ResidualBlock(512, 512)
        
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.outc = OutConv(64, out_channels)
        
    def forward(self, x, stim_trace):
        x_in = torch.cat([x, stim_trace], dim=1)
        
        x1 = self.inc(x_in)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        x = self.bottleneck(x4)
        x = self.up1(x, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        s_out = self.outc(x)
        
        return s_out


