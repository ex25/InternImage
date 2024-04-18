# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from .psp_head import PPM
from .uper_head import UPerHead

@HEADS.register_module()
class My_UPerHead(UPerHead):
    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super(My_UPerHead, self).__init__(pool_scales=(1, 2, 3, 6), **kwargs)

    def forward(self, inputs, real_inputs=None):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        
        if real_inputs is not None:
            real_output = self._forward_feature(real_inputs)
            real_output = self.cls_seg(real_output)
            return output, real_output
        else:
            return output
