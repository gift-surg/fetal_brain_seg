# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import os
import sys
from Demic.net.unet2d_origin import UNet2DOrigin
from Demic.net.unet2d import UNet2D
from Demic.net.pnet import PNet
from Demic.net.pnet_stn_fuse import PNet_STN_DF
from Demic.net.vgg21 import VGG21
class NetFactory(object):
    @staticmethod
    def create(name):
        if name == 'UNet2DOrigin':
            return UNet2DOrigin
        if name == 'UNet2D':
            return UNet2D
        if name == 'PNet':
            return PNet
        if name == 'PNet_STN_DF':
            return PNet_STN_DF
        if name == 'VGG21':
            return VGG21
        print('unsupported network:', name)
        exit()
