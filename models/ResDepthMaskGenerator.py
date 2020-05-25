from collections import OrderedDict
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils import model_zoo
import copy
import numpy as np
from torchvision import utils

class _UpProjection(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super(_UpProjection, self).__init__()

        self.conv1 = nn.Conv2d(num_input_features, num_output_features, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(num_output_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(num_output_features, num_output_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(num_output_features)

        self.conv2 = nn.Conv2d(num_input_features, num_output_features, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(num_output_features)

    def forward(self, x, size):
        x = nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=True)
        x_conv1 = self.relu(self.bn1(self.conv1(x)))
        bran1 = self.bn1_2(self.conv1_2(x_conv1))
        bran2 = self.bn2(self.conv2(x))

        out = self.relu(bran1 + bran2)

        return out

class Encoder(nn.Module):

    def __init__(self, original_model, num_features = 2048):
        super(Encoder, self).__init__()        
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=64, kernel_size=(3, 3), padding=1, bias=False)
        self.bn1 = original_model.bn1
        self.relu = original_model.relu
        self.maxpool = original_model.maxpool

        self.layer1 = original_model.layer1
        self.layer2 = original_model.layer2
        self.layer3 = original_model.layer3
        self.layer4 = original_model.layer4
       

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x_block1 = self.layer1(x)
        x_block2 = self.layer2(x_block1)
        x_block3 = self.layer3(x_block2)
        x_block4 = self.layer4(x_block3)

        return x_block1, x_block2, x_block3, x_block4

class Decoder(nn.Module):

    def __init__(self, num_features = 2048):
        super(Decoder, self).__init__()
        self.conv = nn.Conv2d(num_features, num_features // 2, kernel_size=1, stride=1, bias=False)
        num_features = num_features // 2
        self.bn = nn.BatchNorm2d(num_features)

        self.up1 = _UpProjection(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2

        self.up2 = _UpProjection(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2

        self.up3 = _UpProjection(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2

        self.up4 = _UpProjection(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2


    def forward(self, x_block1, x_block2, x_block3, x_block4):
        x_d0 = F.relu(self.bn(self.conv(x_block4)))
        x_d1 = self.up1(x_d0, [x_block3.size(2), x_block3.size(3)])
        x_d2 = self.up2(x_d1, [x_block2.size(2), x_block2.size(3)])
        x_d3 = self.up3(x_d2, [x_block1.size(2), x_block1.size(3)])
        x_d4 = self.up4(x_d3, [x_block1.size(2)*2, x_block1.size(3)*2])

        return x_d4

class ResDepthMaskGenerator(nn.Module):
    def __init__(self, backbone, num_features, num_out_channels=3, last_conv_inchannels=16):

        super(ResDepthMaskGenerator, self).__init__()

        self.Encoder_backbone = Encoder(backbone)

        self.Decoder_mask = Decoder(num_features)
        self.last_conv_mask = nn.Conv2d(last_conv_inchannels, num_out_channels, kernel_size=5, stride=1, padding=2, bias=True)

        self.Decoder_depth = Decoder(num_features)
        self.last_conv_depth = nn.Conv2d(last_conv_inchannels, num_out_channels, kernel_size=5, stride=1, padding=2, bias=True)


    def forward(self, sample):
        x = torch.cat([sample['image'], sample['bg']], dim=1)

        x_block1, x_block2, x_block3, x_block4 = self.Encoder_backbone(x)

        x_decoder_mask = self.Decoder_mask(x_block1, x_block2, x_block3, x_block4)
        out_mask = self.last_conv_mask(x_decoder_mask)

        x_decoder_depth = self.Decoder_depth(x_block1, x_block2, x_block3, x_block4)
        out_depth = self.last_conv_depth(x_decoder_depth)

        return out_depth, out_mask