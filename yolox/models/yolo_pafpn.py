#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn
from loguru import logger
from .darknet import CSPDarknet
from .network_blocks import BaseConv, CSPLayer, DWConv
from .swin_transformer import SwinTransformer

class YOLOPAFPN(nn.Module):

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        swinFeature_indices=(1, 2, 3),
        in_channels=[192, 384, 768],
        swin_embed_dim = 96,
        swin_depths=[2, 2, 6, 2],
        swin_num_heads=[3, 6, 12, 24],
        swin_window_size = 7,
        swin_drop_path_rate = 0.2,
        swin_pretrained = True, 
        swin_pretrained_type = None,
        swin_pretrained_checkpoint=None,
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        self.backbone = SwinTransformer(
            embed_dim = swin_embed_dim,
            depths = swin_depths,
            num_heads = swin_num_heads,
            window_size = swin_window_size,
            drop_path_rate = swin_drop_path_rate,
        )
        
        if swin_pretrained:
            assert swin_pretrained_type in ["COCO", "ImageNet"], "swin_pretrained_type should be 'COCO' or 'ImageNet'."
            assert swin_pretrained_checkpoint, "swin_pretrained_checkpoint is not given."
            logger.info("Pretrained type: {}, load swin backbone from {}.".format(swin_pretrained_type, swin_pretrained_checkpoint))
            self.load_pretrained(swin_pretrained_type, swin_pretrained_checkpoint) 
   
        self.swinFeature_indices = swinFeature_indices
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

    def load_pretrained(self, swin_pretrained_type, pretrained_checkpoint):
        SwinT_dict = self.backbone.state_dict()
        if swin_pretrained_type == "COCO":
            pretrained_dict = torch.load(pretrained_checkpoint, map_location = 'cpu')['state_dict']
            pretrained_dict = {k.replace('backbone.',''): v for k,v in pretrained_dict.items() if 'backbone' in k}
        elif swin_pretrained_type == "ImageNet":
            pretrained_dict = torch.load(pretrained_checkpoint, map_location = 'cpu')['model']
            pretrained_dict = {k: v for k,v in pretrained_dict.items()}
        logger.info("Used pretrained model parameters:{}".format(pretrained_dict.keys()))
        SwinT_dict.update(pretrained_dict)
        self.backbone.load_state_dict(SwinT_dict,strict=False)
    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone
        out_features = self.backbone(input)
        features = [out_features[i] for i in self.swinFeature_indices]
        [x2, x1, x0] = features

        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = (pan_out2, pan_out1, pan_out0)
        return outputs
