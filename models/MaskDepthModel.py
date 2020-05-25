from collections import OrderedDict
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils import model_zoo
import copy
import numpy as np
from torchvision import utils

from .seg_resnet import *
from .unet_modules import *

class model(nn.Module):
    def __init__(self, Encoder, num_features, block_channel):

        super(model, self).__init__()

        self.E = Encoder
        self.Decoder_mask = modules.D(num_features)
		#self.Decoder_depth = modules.D(num_features)
        #self.MFF = modules.MFF(block_channel)
        self.R = modules.R(block_channel)


    def forward(self, sample):
		x = torch.cat([sample['image'], sample['bg']], dim=1)
        x_block1, x_block2, x_block3, x_block4 = self.E(x)
        x_decoder_mask = self.Decoder_mask(x_block1, x_block2, x_block3, x_block4)
		#x_decoder_depth = self.Decoder_depth(x_block1, x_block2, x_block3, x_block4)
        #x_mff = self.MFF(x_block1, x_block2, x_block3, x_block4,[x_decoder.size(2),x_decoder.size(3)])
        #out = self.R(torch.cat((x_decoder, x_mff), 1))

        return out