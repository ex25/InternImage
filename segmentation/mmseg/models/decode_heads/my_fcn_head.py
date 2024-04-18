# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from .fcn_head import FCNHead

@HEADS.register_module()
class My_FCNHead(FCNHead):
    def __init__(self, *args, **kwargs):
        super(My_FCNHead, self).__init__(*args, **kwargs)

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