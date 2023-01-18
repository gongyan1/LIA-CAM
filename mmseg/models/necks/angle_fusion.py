# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from mmseg.registry import MODELS



# class AngleFusion(nn.Module):
#     def __init__(self, size, channels, step=[4, 2, 1], num_heads=1) -> None:
#         super().__init__()
#         assert isinstance(step, list)
#         ag2vec = []
#         len = size[0]*size[1]
#         st = 1
#         for i in step:
#             ag2vec.append(nn.Linear(st, len//i))
#             ag2vec.append(nn.ReLU())
#             st = len // i
#         self.ag2vec = nn.Sequential(*ag2vec)
#         self.gamma = nn.Parameter(torch.zeros(1))
#         self.num_heads = num_heads
#         self.to_mutiheads = nn.Linear(len, len*num_heads)
#         self.reduction_channels = nn.Conv2d(num_heads*channels, channels, kernel_size=1)
#     def forward(self, featuremap, angle):
#         # featuremap:b * c * h * w
#         # angle:b * 1
#         # HACK:Maybe there are better way to move device
#         angle = angle.to(featuremap.device)
#         featuremap_cp = featuremap
#         batch_size, channel, h, w = featuremap.shape
#         angle_vec = self.ag2vec(angle)
#         angle_map = angle_vec.reshape(batch_size, 1, w, h)
#         # muti heads
#         featuremap = featuremap.reshape(batch_size*channel, h*w)
#         featuremap = self.to_mutiheads(featuremap)
#         featuremap = featuremap.reshape(batch_size*channel*self.num_heads, h, w)
#         angle_map = angle_map.repeat(1, channel*self.num_heads, 1, 1)
#         angle_map = angle_map.reshape(batch_size*channel*self.num_heads, w, h)
#         # fus_weight
#         fus_weight = torch.bmm(angle_map, featuremap)
#         # softmax 0 
#         fus_weight = F.softmax(fus_weight, dim=1)
#         fus_weight = fus_weight / torch.sqrt(torch.tensor(w))
#         # softmax 1
#         # fus_weight = fus_weight.reshape((batch_size*channel, -1))
#         # fus_weight = F.softmax(fus_weight, dim=1)
#         # fus_weight = fus_weight.reshape((batch_size*channel, w, w))
#         # fusion
#         fusion = torch.bmm(featuremap, fus_weight)
#         fusion = fusion.reshape(batch_size, channel*self.num_heads, h, w)
#         fusion = self.reduction_channels(fusion)
        
#         return featuremap_cp + self.gamma * fusion




class AngleFusion(nn.Module):
    def __init__(self, size, channels, step=[4, 2, 1], num_heads=1) -> None:
        super().__init__()
        assert isinstance(step, list)
        ag2vec = []
        len = size[0]*size[1]
        st = 1
        for i in step:
            ag2vec.append(nn.Linear(st, len//i))
            ag2vec.append(nn.ReLU())
            st = len // i
        self.ag2vec = nn.Sequential(*ag2vec)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.num_heads = num_heads
        self.to_mutiheads = nn.Linear(len, len*num_heads)
        self.reduction_channels = nn.Conv2d(num_heads*channels, channels, kernel_size=1)
    def forward(self, featuremap, angle):
        # featuremap:b * c * h * w
        # angle:b * 1
        # HACK:Maybe there are better way to move device
        angle = angle.to(featuremap.device)
        featuremap_cp = featuremap
        batch_size, channel, h, w = featuremap.shape
        angle_vec = self.ag2vec(angle)
        angle_map = angle_vec.reshape(batch_size, 1, w, h)
        # muti heads
        featuremap = featuremap.reshape(batch_size*channel, h*w)
        featuremap = self.to_mutiheads(featuremap)
        featuremap = featuremap.reshape(batch_size*channel*self.num_heads, h, w)
        angle_map = angle_map.repeat(1, channel*self.num_heads, 1, 1)
        angle_map = angle_map.reshape(batch_size*channel*self.num_heads, w, h)
        # fus_weight
        fus_weight = torch.bmm(angle_map, featuremap)
        # softmax 0 
        fus_weight = F.softmax(fus_weight, dim=1)
        fus_weight = fus_weight / torch.sqrt(torch.tensor(w))
        # softmax 1
        # fus_weight = fus_weight.reshape((batch_size*channel, -1))
        # fus_weight = F.softmax(fus_weight, dim=1)
        # fus_weight = fus_weight.reshape((batch_size*channel, w, w))
        # fusion
        fusion = torch.bmm(featuremap, fus_weight)
        fusion = fusion.reshape(batch_size, channel*self.num_heads, h, w)
        fusion = self.reduction_channels(fusion)
        
        return featuremap_cp + self.gamma * fusion



@MODELS.register_module()
class Angle_Fusion_Neck(nn.Module):
    def __init__(self, hw_list, channels_list, step_list, index_list, num_head_list):
        super().__init__()
        assert len(hw_list) == len(index_list) == len(num_head_list) == len(channels_list)
        self.index_list = index_list
        self.num_head_list = num_head_list
        self.fusion_modules = dict()
        for i, index in enumerate(self.index_list):
            self.fusion_modules[str(index)] = AngleFusion(hw_list[i], channels_list[i], step_list, num_head_list[i])
        
    def forward(self, featuremap, data_samples:SampleList):
        featuremap = list(featuremap)
        angle = torch.tensor([sample.get('angle') for sample in data_samples], dtype=torch.float32)
        angle = angle.reshape(-1, 1)
        for index in self.index_list:
            # HACK:Maybe there are better way to move device
            net = self.fusion_modules[str(index)].to(featuremap[index].device)
            featuremap[index] = net(featuremap[index], angle)
        return tuple(featuremap)
        