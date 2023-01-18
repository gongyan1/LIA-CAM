# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Scale
from torch import Tensor, nn

from mmseg.registry import MODELS
from mmseg.utils import SampleList, add_prefix
from ..utils import SelfAttentionBlock as _SelfAttentionBlock
from .decode_head import BaseDecodeHead

from mmcv.cnn import ConvModule, build_activation_layer, build_norm_layer

class DeconvModule(nn.Module):
    """Deconvolution upsample module in decoder for UNet (2X upsample).

    This module uses deconvolution to upsample feature map in the decoder
    of UNet.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict | None): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
        kernel_size (int): Kernel size of the convolutional layer. Default: 4.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 with_cp=False,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 *,
                 kernel_size=4,
                 scale_factor=2):
        super().__init__()

        assert (kernel_size - scale_factor >= 0) and\
               (kernel_size - scale_factor) % 2 == 0,\
               f'kernel_size should be greater than or equal to scale_factor '\
               f'and (kernel_size - scale_factor) should be even numbers, '\
               f'while the kernel size is {kernel_size} and scale_factor is '\
               f'{scale_factor}.'

        stride = scale_factor
        padding = (kernel_size - scale_factor) // 2
        self.with_cp = with_cp
        deconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding)

        norm_name, norm = build_norm_layer(norm_cfg, out_channels)
        activate = build_activation_layer(act_cfg)
        self.deconv_upsamping = nn.Sequential(deconv, norm, activate)

    def forward(self, x):
        """Forward function."""

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(self.deconv_upsamping, x)
        else:
            out = self.deconv_upsamping(x)
        return out


class PreAngle(nn.Module):
    def __init__(self, in_channels, num_layers) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.layer = nn.Sequential()
        for i in range(num_layers):
            self.layer.add_module('block{}'.format(i), nn.Sequential(
                nn.Conv2d(self.in_channels, self.in_channels//2, kernel_size=3, stride=1, padding=1), 
                nn.BatchNorm2d(self.in_channels//2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)))
            self.in_channels = self.in_channels // 2
        
        self.out = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)), 
            nn.Flatten(),
            nn.Linear(self.in_channels, self.in_channels//2),
            nn.ReLU(),
            nn.Linear(self.in_channels//2, 1))
    
    def forward(self, featuremap):
        featuremap = self.layer(featuremap)
        out = self.out(featuremap)
        return out


class PAM(_SelfAttentionBlock):
    """Position Attention Module (PAM)

    Args:
        in_channels (int): Input channels of key/query feature.
        channels (int): Output channels of key/query transform.
    """

    def __init__(self, in_channels, channels):
        super().__init__(
            key_in_channels=in_channels,
            query_in_channels=in_channels,
            channels=channels,
            out_channels=in_channels,
            share_key_query=False,
            query_downsample=None,
            key_downsample=None,
            key_query_num_convs=1,
            key_query_norm=False,
            value_out_num_convs=1,
            value_out_norm=False,
            matmul_norm=False,
            with_out=False,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None)

        self.gamma = Scale(0)

    def forward(self, x):
        """Forward function."""
        out = super().forward(x, x)

        out = self.gamma(out) + x
        return out


class CAM(nn.Module):
    """Channel Attention Module (CAM)"""

    def __init__(self):
        super().__init__()
        self.gamma = Scale(0)

    def forward(self, x):
        """Forward function."""
        batch_size, channels, height, width = x.size()
        proj_query = x.view(batch_size, channels, -1)
        proj_key = x.view(batch_size, channels, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(
            energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = F.softmax(energy_new, dim=-1)
        proj_value = x.view(batch_size, channels, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(batch_size, channels, height, width)

        out = self.gamma(out) + x
        return out


@MODELS.register_module()
class Angle_DAHead(BaseDecodeHead):
    """Dual Attention Network for Scene Segmentation.

    This head is the implementation of `DANet
    <https://arxiv.org/abs/1809.02983>`_.

    Args:
        pam_channels (int): The channels of Position Attention Module(PAM).
    """

    def __init__(self, pam_channels, 
                 upsample_scale_factor_list=[2, 2],
                 num_pre_angle_layers=2, 
                 with_angle_loss=False,
                 channels_list = [],
                 cout=64,
                 **kwargs):

        super().__init__(**kwargs)
        self.pam_channels = pam_channels
        self.pam_in_conv = ConvModule(
            self.in_channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.pam = PAM(self.channels, pam_channels)
        self.pam_out_conv = ConvModule(
            self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.pam_conv_seg = nn.Conv2d(
            self.channels, self.num_classes, kernel_size=1)

        self.cam_in_conv = ConvModule(
            self.in_channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.cam = CAM()
        self.cam_out_conv = ConvModule(
            self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.cam_conv_seg = nn.Conv2d(
            self.channels, self.num_classes, kernel_size=1)

        upsamples = []
        # for i, scale_factor in enumerate(upsample_scale_factor_list):
        #     upsamples.append(DeconvModule(self.channels, self.channels, scale_factor=scale_factor))
        upsamples.append(nn.ConvTranspose2d(self.channels, self.channels, kernel_size=16, stride=8, padding=4))
        self.upsamples = nn.Sequential(*upsamples)

        self.with_angle_loss = with_angle_loss
        if with_angle_loss:
            assert num_pre_angle_layers != None
            self.pre_angle = PreAngle(self.channels, num_pre_angle_layers)
            self.loss_angle = nn.MSELoss()

        self.channels_list = channels_list
        self.cout = cout
        self.score_net = nn.ModuleList()
        self.deconv_net = nn.ModuleList()
        for i in self.channels_list[::-1]:
            # self.score_net.append(ConvModule(
            #     i, 
            #     self.cout, 
            #     kernel_size=1, 
            #     stride=1, 
            #     conv_cfg=self.conv_cfg,
            #     norm_cfg=self.norm_cfg,
            #     act_cfg=self.act_cfg))
            self.score_net.append(nn.Conv2d(i, self.cout, 1, stride=1, bias=False))
            if i > 0:
                self.deconv_net.append(nn.ConvTranspose2d(self.cout, self.cout, 4, 2, padding=1, bias=False))

        #self.pre_head = ConvModule(self.cout, self.cout*8, 1, 1)
        self.pre_head = nn.ConvTranspose2d(self.cout, self.cout*8, 1, 1)
        self.conv_seg = nn.Conv2d(128, self.out_channels, kernel_size=1)
        
    def pam_cls_seg(self, feat):
        """PAM feature classification."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        #feat = self.upsamples(feat)
        output = self.pam_conv_seg(feat)
        return output

    def cam_cls_seg(self, feat):
        """CAM feature classification."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        #feat = self.upsamples(feat)
        output = self.cam_conv_seg(feat)
        return output
    
    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        feat = self.upsamples(feat)
        output = self.conv_seg(feat)
        return output
    
    def _forward_preangle(self, seg_logits):
        pre_angle = self.pre_angle(seg_logits)
        return pre_angle

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            # assert isinstance(in_channels, (list, tuple))
            # assert isinstance(in_index, (list, tuple))
            # assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            # assert isinstance(in_channels, int)
            # assert isinstance(in_index, int)
            self.in_channels = in_channels

      
    def forward(self, inputs):
        """Forward function."""
        inputs = self._transform_inputs(inputs)

        for i,input in enumerate(inputs[::-1]):
            if i > 0:
                deconv = self.deconv_net[i-1](score)        
            score = self.score_net[i](input)
            if i > 0:
                score = score + deconv
        x = self.pre_head(score)
        
        pam_feat = self.pam_in_conv(x)
        pam_feat = self.pam(pam_feat)
        pam_feat = self.pam_out_conv(pam_feat)
        pam_out = self.pam_cls_seg(pam_feat)

        cam_feat = self.cam_in_conv(x)
        cam_feat = self.cam(cam_feat)
        cam_feat = self.cam_out_conv(cam_feat)
        cam_out = self.cam_cls_seg(cam_feat)

        feat_sum = pam_feat + cam_feat
        
        pam_cam_out = self.cls_seg(feat_sum)
        
        if self.with_angle_loss and self.training:
            pre_angle = self._forward_preangle(pam_cam_out)
            return (pam_cam_out, pam_out, cam_out), pre_angle
        else:
            return (pam_cam_out, pam_out, cam_out)

    def predict(self, inputs, batch_img_metas: List[dict], test_cfg,
                **kwargs) -> List[Tensor]:
        """Forward function for testing, only ``pam_cam`` is used."""
        seg_logits = self.forward(inputs)[0]
        return self.predict_by_feat(seg_logits, batch_img_metas, **kwargs)

    def loss_by_feat(self, seg_logit: Tuple[Tensor],
                     batch_data_samples: SampleList, **kwargs) -> dict:
        """Compute ``pam_cam``, ``pam``, ``cam`` loss."""
        pam_cam_seg_logit, pam_seg_logit, cam_seg_logit = seg_logit
        loss = dict()
        loss.update(
            add_prefix(
                super().loss_by_feat(pam_cam_seg_logit, batch_data_samples),
                'pam_cam'))
        loss.update(
            add_prefix(super().loss_by_feat(pam_seg_logit, batch_data_samples),
                       'pam'))
        loss.update(
            add_prefix(super().loss_by_feat(cam_seg_logit, batch_data_samples),
                       'cam'))
        return loss

    def loss_by_angle(self, pre_angle, angle):
        angle = angle.to(pre_angle.device)
        loss = dict() 
        loss['loss_mse'] = self.loss_angle(pre_angle, angle).mean()
        return loss
    
    def loss(self, inputs: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg) -> dict:
        """Forward function for training.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_data_samples (list[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `img_metas` or `gt_semantic_seg`.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        losses = dict()
        if self.with_angle_loss:
            angle = torch.tensor([sample.angle for sample in batch_data_samples], dtype=torch.float32)
            angle = angle.reshape(-1, 1) 
            seg_logits, pre_angle = self.forward(inputs)
            loss_by_angle = self.loss_by_angle(pre_angle, angle)
            losses.update(loss_by_angle)
        else:
            seg_logits = self.forward(inputs)
        loss_by_feat = self.loss_by_feat(seg_logits, batch_data_samples)
        losses.update(loss_by_feat)
        return losses