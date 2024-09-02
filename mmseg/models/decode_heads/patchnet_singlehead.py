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
                 conv_next=False,
                 num_layer=2,
                 **kwargs):
        super().__init__( **kwargs)
        self.input_dim = self.channels 
        self.interpolate_mode = interpolate_mode
        self.conv_kernel_size = conv_kernel_size
        self.conv_next=conv_next
        self.conv_seg = None

        if self.conv_next:
            layer = []
            layer.append(nn.Conv2d(self.input_dim, self.input_dim//2, kernel_size=self.conv_kernel_size, padding=self.conv_kernel_size//2, stride = 2))
            # layer.append(nn.SyncBatchNorm(self.input_dim//2))
            layer.append(nn.ReLU(inplace=True))
            layer.extend([Block(dim=self.input_dim//2, kernel_size=conv_kernel_size) for _ in range(num_layer)])
            # layer.append(nn.Conv2d(self.input_dim//2, self.input_dim//2, kernel_size=1, padding=1))
            # layer.append(nn.SyncBatchNorm(self.input_dim//2))
            # layer.append(nn.ReLU(inplace=True))
            layer.append(nn.Conv2d(self.input_dim//2, self.num_classes, kernel_size=1))
            self.seg_head = nn.Sequential(*layer)
        elif self.conv_kernel_size == 1:
            self.seg_head = nn.Sequential(
                nn.Conv2d(self.input_dim, self.input_dim//2, kernel_size=self.conv_kernel_size),
                nn.SyncBatchNorm(self.input_dim//2),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.input_dim//2, self.num_classes, kernel_size=1)
                )
        # elif self.conv_kernel_size == 3:
        #     self.seg_head = nn.Sequential(
        #         nn.Conv2d(self.input_dim, self.input_dim//2, kernel_size=self.conv_kernel_size, padding=1),
        #         nn.SyncBatchNorm(self.input_dim//2),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(self.input_dim//2, self.num_classes, kernel_size=1)
        #         )
        elif self.conv_kernel_size == 3:
            self.seg_head = nn.Sequential(
                nn.Conv2d(self.input_dim, self.input_dim//2, kernel_size=self.conv_kernel_size, padding=1),
                nn.SyncBatchNorm(self.input_dim//2),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.input_dim//2, self.input_dim//2, kernel_size=self.conv_kernel_size, padding=1),
                nn.SyncBatchNorm(self.input_dim//2),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.input_dim//2, self.num_classes, kernel_size=1)
            )
    
    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        x = inputs[-1]
        if self.conv_next:
            x = F.interpolate(x, size=(32, 32),mode=self.interpolate_mode)
        else:
            x = F.interpolate(x, size=(16, 16),mode=self.interpolate_mode)
        if self.seg_head is not None:
            seg_out = self.seg_head(x)
        else:
            seg_out = None
        return seg_out

class GRN(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super(GRN, self).__init__()
        self.dim = dim
        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def init_fn(self, shape, fill_value):
        return torch.full(shape, fill_value)

    def forward(self, inputs, mask=None):
        x = inputs
        if mask is not None:
            x = x * (1. - mask)
        Gx = torch.sqrt(torch.sum(torch.pow(x, 2), dim=(1, 2), keepdim=True) + self.eps)
        Nx = Gx / (torch.mean(Gx, dim=-1, keepdim=True) + self.eps)
        return self.gamma * (Nx * inputs) + self.beta + inputs

class Block(nn.Module):
    """ ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_path=0., kernel_size =7):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim) # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = input + x
        return x