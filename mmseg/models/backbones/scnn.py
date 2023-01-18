# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModel
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)

from mmseg.registry import MODELS


class _SCNN(nn.Module):
    def __init__(self, 
                 channels=128):
        super(_SCNN, self).__init__()
        self.conv_d = nn.Conv2d(channels, channels, (1, 9), padding=(0, 4), bias=False)
        self.conv_u = nn.Conv2d(channels, channels, (1, 9), padding=(0, 4), bias=False)
        self.conv_r = nn.Conv2d(channels, channels, (9, 1), padding=(4, 0), bias=False)
        self.conv_l = nn.Conv2d(channels, channels, (9, 1), padding=(4, 0), bias=False)

    def forward(self, x):
        x = x.clone()
        for i in range(1, x.shape[2]):
            x[..., i:i+1, :].add_(F.relu(self.conv_d(x[..., i-1:i, :])))

        for i in range(x.shape[2] - 2, 0, -1):
            x[..., i:i+1, :].add_(F.relu(self.conv_u(x[..., i+1:i+2, :])))

        for i in range(1, x.shape[3]):
            x[..., i:i+1].add_(F.relu(self.conv_r(x[..., i-1:i])))

        for i in range(x.shape[3] - 2, 0, -1):
            x[..., i:i+1].add_(F.relu(self.conv_l(x[..., i+1:i+2])))
        return x

@MODELS.register_module()
class SCNN(nn.Module):
    def __init__(self,
                 backbone_cfg,
                 in_index_list,
                 channels_list) -> None:
        super().__init__()
        self.backbone = MODELS.build(backbone_cfg)
        self.in_index_list = in_index_list
        self.channels_list = channels_list
        self.scnn_modules = dict()
        for i, index in enumerate(self.in_index_list):
            self.scnn_modules[str(index)] = _SCNN(channels=self.channels_list[i])
    
    def forward(self, x):
        featuremap = self.backbone(x)
        featuremap = list(featuremap)
        for index in self.in_index_list:
            net = self.scnn_modules[str(index)].to(featuremap[index].device)
            featuremap[index] = net(featuremap[index])
        return tuple(featuremap)

        
        
        
        