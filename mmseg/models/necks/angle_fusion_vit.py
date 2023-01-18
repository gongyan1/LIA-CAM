# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmseg.models.backbones.vit import TransformerEncoderLayer
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from mmseg.registry import MODELS


class CATransformer(TransformerEncoderLayer):
    def forward(self, q, k_and_v):
        x = self.attn(query=self.norm1(q), key=self.norm1(k_and_v), value=self.norm1(k_and_v), identity=k_and_v)
        x = self.ffn(self.norm2(x), identity=x)
        return x


class AngleFusion(nn.Module):
    def __init__(self, size, channels, step=[4, 2, 1], num_heads=1) -> None:
        super().__init__()
        self.ang2h = nn.Linear(1, size[0])
        self.ang2w = nn.Linear(1, size[1])
        self.ang2c = nn.Linear(1, channels)
        
        self.catransformer = CATransformer(channels, num_heads=num_heads, feedforward_channels=2 * channels)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.num_heads = num_heads

    def forward(self, img_feature_map, angle):
        # featuremap:b * c * h * w
        # angle:b * 1
        # HACK:Maybe there are better way to move device
        b, c, h, w = img_feature_map.shape
        img_feature_map_copy = img_feature_map
        angle = angle.to(img_feature_map.device)
        
        ang2h = self.ang2h(angle).unsqueeze(2)
        ang2w = self.ang2w(angle).unsqueeze(1)
        ang2c = self.ang2c(angle).reshape(b, 1, 1, c)
        ang2c = ang2c.repeat(1, h, w, 1)

        hw = torch.bmm(ang2h, ang2w).unsqueeze(3)
        hw = hw.repeat(1, 1, 1, c)
        angle_feature_map = torch.mul(hw, ang2c)
        angle_feature_map = angle_feature_map.transpose(1, 3)
        
        img_feature_map = img_feature_map.flatten(2)  # b * c * hw
        img_feature_map = img_feature_map.transpose(-1, -2)  # b * hw * c
        angle_feature_map = angle_feature_map.flatten(2)  # b * c * hw
        angle_feature_map = angle_feature_map.transpose(-1, -2)  # b * hw * c

        fusion_output = self.catransformer(angle_feature_map, img_feature_map)  # b * hw * c
        fusion_output = fusion_output.permute(0, 2, 1)  # b * c * hw
        fusion_output = fusion_output.contiguous().view(b, c, h, w)  # b * c * h * w

        return img_feature_map_copy + self.gamma * fusion_output
        


@MODELS.register_module()
class Angle_Fusion_Vit_Neck(nn.Module):
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
        