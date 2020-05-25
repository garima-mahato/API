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

class Quiz9_DNN_Net(nn.Module):
    def __init__(self, dropout_value):
        super(Quiz9_DNN_Net, self).__init__()

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
        
        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32+64, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),
        ) # input_size = 32x32x128, output_size = 32x32x32, RF = 7x7
        self.pool1 = nn.MaxPool2d(2, 2) # input_size = 32x32x32, output_size = 16x16x32, RF = 8x8

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
			nn.Conv2d(in_channels=32+64+32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
			nn.ReLU(),
			nn.BatchNorm2d(64),
			nn.Dropout(self.dropout_value)
		) # input_size = 16x16x32, output_size = 16x16x64, RF = 12x12
		
        self.convblock5 = nn.Sequential(
			nn.Conv2d(in_channels=32+64+32+64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False),
			nn.ReLU(),
			nn.BatchNorm2d(128),
			nn.Dropout(self.dropout_value)
        )

        # TRANSITION BLOCK 2
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=32+64+32+64+128, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),
        ) # input_size = 16x16x128, output_size = 16x16x64, RF = 16x16
        self.pool2 = nn.MaxPool2d(2, 2) # input_size = 16x16x64, output_size = 8x8x64, RF = 18x18

        # CONVOLUTION BLOCK 3
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=64+128+32, out_channels=64, kernel_size=(3, 3), padding=2, dilation=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(self.dropout_value)
        ) # input_size = 8x8x64, output_size = 8x8x128, RF = [18+(5-1)*4] = 34x34
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=64+128+32+64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        ) # input_size = 8x8x128, output_size = 8x8x256, RF = [34+(3-1)*4] = 42x42

        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=64+128+32+64+64, out_channels=32, kernel_size=(1, 1), padding=0, bias=False)
        ) # input_size = 4x4x64, output_size = 4x4x128, RF = 46+(3-1)*8 = 62x62
                        
        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=8)
        ) # input_size = 8x8x32, output_size = 1x1x32, RF = 78+(4-1)*8 = 102x102

        self.convblock10 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)
        ) # input_size = 1x1x32, output_size = 1x1x10, RF = 102x102 


    def forward(self, x):
        x1 = self.convblock1(x)
        x2 = self.convblock2(x1)
        x3 = self.convblock3(torch.cat([x1, x2], 1))
        x4 = self.pool1(torch.cat([x1, x2, x3], 1))

        x5 = self.convblock4(x4)
        x6 = self.convblock5(torch.cat([x4, x5], 1))
        x7 = self.convblock6(torch.cat([x4, x5, x6], 1))
        x8 = self.pool2(torch.cat([x5, x6, x7], 1))

        x9 = self.convblock7(x8)
        x10 = self.convblock8(torch.cat([x8, x9], 1))
        x11 = self.convblock9(torch.cat([x8, x9, x10], 1))

        x12 = self.gap(x11)     
        x13 = self.convblock10(x12)

        x13 = x13.view(-1, 10)
        return F.log_softmax(x13, dim=-1)