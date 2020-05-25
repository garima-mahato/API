from __future__ import print_function
import math
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from torchsummary import summary

class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class Net(nn.Module):
    def __init__(self, dropout_value):
        super(Net, self).__init__()

        self.dropout_value = dropout_value
        self.num_of_channels = 3

        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=self.num_of_channels, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(self.dropout_value)
        ) # input_size = 32x32x3, output_size = 32x32x32, RF = 3x3

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(self.dropout_value)
        ) # input_size = 32x32x32, output_size = 32x32x64, RF = 5x5
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(self.dropout_value)
        ) # input_size = 32x32x64, output_size = 32x32x128, RF = 7x7

        # TRANSITION BLOCK 1
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),
        ) # input_size = 32x32x128, output_size = 32x32x32, RF = 7x7
        self.pool1 = nn.MaxPool2d(2, 2) # input_size = 32x32x32, output_size = 16x16x32, RF = 8x8

        # CONVOLUTION BLOCK 2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(self.dropout_value)
        ) # input_size = 16x16x32, output_size = 16x16x64, RF = 12x12

        self.convblock6 = SeparableConv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=1, dilation=1, bias=False)
        # input_size = 16x16x64, output_size = 16x16x128, RF = 16x16

        # TRANSITION BLOCK 2
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1), padding=0, bias=False),
        ) # input_size = 16x16x128, output_size = 16x16x64, RF = 16x16
        self.pool2 = nn.MaxPool2d(2, 2) # input_size = 16x16x64, output_size = 8x8x64, RF = 18x18

        # CONVOLUTION BLOCK 3
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=2, dilation=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(self.dropout_value)
        ) # input_size = 8x8x64, output_size = 8x8x128, RF = [18+(5-1)*4] = 34x34
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout(self.dropout_value)
        ) # input_size = 8x8x128, output_size = 8x8x256, RF = [34+(3-1)*4] = 42x42

        # TRANSITION BLOCK 3
        self.convblock10 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(1, 1), padding=0, bias=False),
        ) # input_size = 8x8x256, output_size = 8x8x64, RF = 42x42
        self.pool3 = nn.MaxPool2d(2, 2) # input_size = 8x8x64, output_size = 4x4x64, RF = 42+(2-1)*4 = 46x46

        # CONVOLUTION BLOCK 4
        self.convblock11 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(self.dropout_value)
        ) # input_size = 4x4x64, output_size = 4x4x128, RF = 46+(3-1)*8 = 62x62
        self.convblock12 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout(self.dropout_value)
        ) # input_size = 4x4x128, output_size = 4x4x256, RF = 62+(3-1)*8 = 78x78
                
        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=4)
        ) # input_size = 4x4x256, output_size = 1x1x256, RF = 78+(4-1)*8 = 102x102

        self.convblock13 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(1, 1), padding=0, bias=False)
        ) # input_size = 1x1x256, output_size = 1x1x64, RF = 102x102
        self.convblock14 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1), padding=0, bias=False)
        ) # input_size = 1x1x64, output_size = 1x1x32, RF = 102x102
        self.convblock15 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)
        ) # input_size = 1x1x32, output_size = 1x1x10, RF = 102x102 


    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.pool1(x)

        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.pool2(x)

        x = self.convblock8(x)
        x = self.convblock9(x)
        x = self.convblock10(x)
        x = self.pool3(x)

        x = self.convblock11(x)
        x = self.convblock12(x)

        x = self.gap(x)        
        x = self.convblock13(x)
        x = self.convblock14(x)
        x = self.convblock15(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)