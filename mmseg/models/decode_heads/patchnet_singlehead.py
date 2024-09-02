# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule
from mmcv.cnn import build_norm_layer
from mmseg.models.decode_heads.patch_singlehead_decode_head import patch_singlehead_BaseDecodeHead
from mmseg.registry import MODELS
from ..utils import resize
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)

@MODELS.register_module()
class PatchnetSingleHead(patch_singlehead_BaseDecodeHead):
    """The all mlp Head of segformer.

    This head is the implementation of
    `Segformer <https://arxiv.org/abs/2105.15203>` _.

    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
    """

    def __init__(self,
                 seg_head=None,
                 interpolate_mode='bilinear',
                 conv_kernel_size=1,
                 **kwargs):
        super().__init__( **kwargs)
        self.input_dim = self.channels 
        self.interpolate_mode = interpolate_mode
        self.conv_kernel_size = conv_kernel_size
        self.conv_seg = None
        if self.conv_kernel_size == 1:
            self.seg_head = nn.Sequential(
                nn.Conv2d(self.input_dim, self.input_dim//2, kernel_size=self.conv_kernel_size),
                nn.SyncBatchNorm(self.input_dim//2),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.input_dim//2, self.num_classes, kernel_size=1)
                )
        elif self.conv_kernel_size == 3:
            self.seg_head = nn.Sequential(
                nn.Conv2d(self.input_dim, self.input_dim//2, kernel_size=self.conv_kernel_size, padding=1),
                nn.SyncBatchNorm(self.input_dim//2),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.input_dim//2, self.num_classes, kernel_size=1)
                )
        elif self.conv_kernel_size == 'multi':
            self.conv_kernel_size = 3
            self.seg_head = nn.Sequential(
                nn.Conv2d(self.input_dim, self.input_dim//2, kernel_size=self.conv_kernel_size, padding=1),
                nn.SyncBatchNorm(self.input_dim//2),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.input_dim//2, self.num_classes, kernel_size=1)
                )
            self.seg_head_dilated = nn.Sequential(
                nn.Conv2d(self.input_dim, self.input_dim//2, kernel_size=self.conv_kernel_size, padding=2, dilation=2),
                nn.SyncBatchNorm(self.input_dim//2),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.input_dim//2, self.num_classes, kernel_size=1)
                )
            self.seg_head_dilatedv2 = nn.Sequential(
                nn.Conv2d(self.input_dim, self.input_dim//2, kernel_size=self.conv_kernel_size, padding=3, dilation=3),
                nn.SyncBatchNorm(self.input_dim//2),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.input_dim//2, self.num_classes, kernel_size=1)
                )
    
    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        x = inputs[-1]
        x = F.interpolate(x, size=(16, 16),mode=self.interpolate_mode)
        if self.seg_head is not None:
            seg_out = self.seg_head(x)
            seg_out_dilated = self.seg_head_dilated(x)
            seg_out_dilatedv2 = self.seg_head_dilatedv2(x)
            seg_out = seg_out + seg_out_dilated + seg_out_dilatedv2
        else:
            seg_out = None

        return seg_out
