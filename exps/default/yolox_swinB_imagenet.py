#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        
        #  model config  #
        self.num_classes = 1
        self.depth = 1.00
        self.width = 1.00

        #  swin config #
        self.swin_embed_dim = 128
        self.swin_depths = [2, 2, 18, 2]
        self.swin_num_heads = [4, 8, 16, 32]
        self.swin_window_size = 7
        self.swin_out_channels = [128, 256, 512, 1024]
        self.swin_drop_path_rate = 0.3
        self.swin_pretrained = True
        self.swin_pretrained_type = "ImageNet" # 'ImageNet' or 'COCO'
        self.swin_pretrained_checkpoint = "./pretrained/swin_base_patch4_window7_224_22k.pth"
    
        #  transform config  #
        self.degrees = 120.0

        #  dataloader config #
        self.data_num_workers = 2
        self.input_size = (320, 320)
        self.random_size = (10, 10)
        self.data_dir = "/datasets/coco"
        self.train_ann = "detection_train2017.json"
        self.val_ann = "detection_val2017.json"

        # training config #
        self.basic_lr_per_img = 0.01 / 64.0
        self.eval_interval = 5

        # testing config #
        self.test_size = (320, 320)
        self.test_conf = 0.2
        self.nmsthre = 0.2
