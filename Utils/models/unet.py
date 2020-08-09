# Reference https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
from PIL import Image
import cv2
import numpy as np
import torch
import os
from tqdm import notebook
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
     
        self.inc = DoubleConv(n_channels, 32) 

        # ---------- For Depth ------------------
        self.down1_depth = Down(32, 32) 
        self.down2_depth = Down(32, 64) 
        self.down3_depth = Down(64, 128) 
        self.down4_depth = Down(128, 256)

        factor = 2 if bilinear else 1

        self.down5_depth = Down(256, 512 // factor)

        self.up1_depth = Up(512, 256 // factor, bilinear) 
        self.up2_depth = Up(256, 128 // factor, bilinear) 
        self.up3_depth = Up(128, 64 // factor, bilinear) 
        self.up4_depth = Up(64, 32 , bilinear)
        self.up5_depth = Up(64, 32 , bilinear)

        self.outc_depth = OutConv(32, n_classes) 

      # -------------- For Mask -------------------
        self.down1_mask = Down(32, 64)
        self.down2_mask = Down(64, 128)
        self.down3_mask = Down(128, 256)

        factor = 2 if bilinear else 1

        self.down4_mask = Down(256, 512 // factor)

        self.up1_mask = Up(512, 256 // factor, bilinear)
        self.up2_mask = Up(256, 128 // factor, bilinear)
        self.up3_mask = Up(128, 64 // factor, bilinear)
        self.up4_mask = Up(64, 32, bilinear)
        self.outc_mask = OutConv(32, n_classes)

    def forward(self, input_img):
        
        x1 = self.inc(input_img)

       
      
        x2_depth = self.down1_depth(x1)
        x3_depth = self.down2_depth(x2_depth)
        x4_depth = self.down3_depth(x3_depth)
        x5_depth = self.down4_depth(x4_depth)
        x6_depth = self.down5_depth(x5_depth)
        
      
        x_depth = self.up1_depth(x6_depth, x5_depth)
        x_depth = self.up2_depth(x_depth, x4_depth)
        x_depth = self.up3_depth(x_depth, x3_depth)
        x_depth = self.up4_depth(x_depth, x2_depth)
        x_depth = self.up5_depth(x_depth, x1)
        
        result_depth = self.outc_depth(x_depth)
        


        #----------------------------------------------
      
        
        x2_mask = self.down1_mask(x1)
        x3_mask = self.down2_mask(x2_mask)
        x4_mask = self.down3_mask(x3_mask)
        x5_mask = self.down4_mask(x4_mask)
        
        x_mask = self.up1_mask(x5_mask, x4_mask)
        x_mask = self.up2_mask(x_mask, x3_mask)
        x_mask = self.up3_mask(x_mask, x2_mask)
        x_mask = self.up4_mask(x_mask, x1)
       
        result_mask = self.outc_mask(x_mask)
      
        
        return  result_mask,result_depth
      



