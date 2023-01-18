# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModel
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)

from mmseg.registry import MODELS

# class AngleFusion(nn.Module):
#     def __init__(self, size, channels, step=[4, 2, 1]) -> None:
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

#     def forward(self, featuremap, angle):
#         # featuremap:b*c*h*w
#         # angle:b*1
#         # HACK:Maybe there are better way to move device
#         angle = angle.to(featuremap.device)
#         featuremap_cp = featuremap
#         batch_size, channel, h, w = featuremap.shape
#         angle_vec = self.ag2vec(angle)
#         angle_map = angle_vec.reshape(batch_size, 1, w, h)
#         angle_map = angle_map.repeat(1, channel, 1, 1)
#         featuremap = featuremap.reshape(batch_size*channel, h, w)
#         angle_map = angle_map.reshape(batch_size*channel, w, h)
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
#         fusion = fusion.reshape(batch_size, channel, h, w)
        
#         return featuremap_cp + self.gamma * fusion



# 通道注意力机制（FC）
class Channel_Attention_Module_FC(nn.Module):
    def __init__(self, channels, ratio):
        super(Channel_Attention_Module_FC, self).__init__()
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_pooling = nn.AdaptiveMaxPool2d(1)
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features = channels, out_features = channels // ratio, bias = False),
            nn.ReLU(),
            nn.Linear(in_features = channels // ratio, out_features = channels, bias = False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        avg_x = self.avg_pooling(x).view(b, c)
        max_x = self.max_pooling(x).view(b, c)
        v = self.fc_layers(avg_x) + self.fc_layers(max_x)
        v = self.sigmoid(v).view(b, c, 1, 1)
        v = x * v
        return v

# 通道注意力机制（conv）
class Channel_Attention_Module_Conv(nn.Module):
    def __init__(self, channels, gamma = 2, b = 1):
        super(Channel_Attention_Module_Conv, self).__init__()
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_pooling = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size = kernel_size, padding = (kernel_size - 1) // 2, bias = False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_x = self.avg_pooling(x)
        max_x = self.max_pooling(x)
        avg_out = self.conv(avg_x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        max_out = self.conv(max_x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        v = self.sigmoid(avg_out + max_out)
        return x * v

# 空间注意力机制
class Spatial_Attention_Module(nn.Module):
    def __init__(self, k: int):
        super(Spatial_Attention_Module, self).__init__()
        self.avg_pooling = torch.mean
        self.max_pooling = torch.max
        # In order to keep the size of the front and rear images consistent
        # with calculate, k = 1 + 2p, k denote kernel_size, and p denote padding number
        # so, when p = 1 -> k = 3; p = 2 -> k = 5; p = 3 -> k = 7, it works. when p = 4 -> k = 9, it is too big to use in network
        assert k in [3, 5, 7], "kernel size = 1 + 2 * padding, so kernel size must be 3, 5, 7"
        self.conv = nn.Conv2d(2, 1, kernel_size = (k, k), stride = (1, 1), padding = ((k - 1) // 2, (k - 1) // 2),
                              bias = False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # compress the C channel to 1 and keep the dimensions
        avg_x = self.avg_pooling(x, dim = 1, keepdim = True)
        max_x, _ = self.max_pooling(x, dim = 1, keepdim = True)
        v = self.conv(torch.cat((max_x, avg_x), dim = 1))
        v = self.sigmoid(v)
        return x * v


class CBAMBlock(nn.Module):
    def __init__(self, channel_attention_mode: str, spatial_attention_kernel_size: int, channels: int = None,
                 ratio: int = None, gamma: int = None, b: int = None):
        super(CBAMBlock, self).__init__()
        if channel_attention_mode == "FC":
            assert channels != None and ratio != None and channel_attention_mode == "FC", \
                "FC channel attention block need feature maps' channels, ratio"
            self.channel_attention_block = Channel_Attention_Module_FC(channels = channels, ratio = ratio)
        elif channel_attention_mode == "Conv":
            assert channels != None and gamma != None and b != None and channel_attention_mode == "Conv", \
                "Conv channel attention block need feature maps' channels, gamma, b"
            self.channel_attention_block = Channel_Attention_Module_Conv(channels = channels, gamma = gamma, b = b)
        else:
            assert channel_attention_mode in ["FC", "Conv"], \
                "channel attention block must be 'FC' or 'Conv'"
        self.spatial_attention_block = Spatial_Attention_Module(k = spatial_attention_kernel_size)

    def forward(self, x):
        x = self.channel_attention_block(x)
        x = self.spatial_attention_block(x)
        return x


class AngleFusion(nn.Module):
    def __init__(self, size, channels, step=[4, 2, 1]) -> None:
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
        
        self.attention_m = CBAMBlock("FC", 5, channels = 512, ratio = 9)

    def forward(self, featuremap, angle):
        # featuremap:b*c*h*w
        # angle:b*1
        # HACK:Maybe there are better way to move device
        angle = angle.to(featuremap.device)
        featuremap_cp = featuremap
        batch_size, channel, h, w = featuremap.shape
        angle_vec = self.ag2vec(angle)
        angle_map = angle_vec.reshape(batch_size, 1, w, h)
        angle_map = angle_map.repeat(1, channel, 1, 1)
        featuremap = featuremap.reshape(batch_size*channel, h, w)
        angle_map = angle_map.reshape(batch_size*channel, h, w)
        
        fus = featuremap + angle_map
        fus = fus.reshape(batch_size, channel, h, w)
        fus = self.attention_m(fus)
        
        return fus

@MODELS.register_module()
class Angle_Fusion_Neck_Old(nn.Module):
    def __init__(self, hw_list, channels_list, step_list, index_list):
        super().__init__()
        assert len(hw_list) == len(index_list)  == len(channels_list)
        self.index_list = index_list
        
        self.fusion_modules = dict()
        for i, index in enumerate(self.index_list):
            self.fusion_modules[str(index)] = AngleFusion(hw_list[i], channels_list[i], step_list)
        
    def forward(self, featuremap, data_samples:SampleList):
        featuremap = list(featuremap)
        angle = torch.tensor([sample.get('angle') for sample in data_samples], dtype=torch.float32)
        angle = angle.reshape(-1, 1)
        for index in self.index_list:
            # HACK:Maybe there are better way to move device
            net = self.fusion_modules[str(index)].to(featuremap[index].device)
            featuremap[index] = net(featuremap[index], angle)
        return tuple(featuremap)