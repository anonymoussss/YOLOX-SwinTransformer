# YOLOX with Swin-Transformer backbone

## YOLOX Version
**[0.1.1]** , Aug, 2021

## Introduction
> In short, the content of this repository is yolox with Swin-Transformer as the backbone.
> 简而言之，这个仓库的内容是以swin-transformer为backbone的yolox。

YOLOX is an anchor-free version of YOLO, with a simpler design but better performance. I rewrote the version with Swin-Transformer as backbone following Swin-Transformer-Object-Detection(https://github.com/SwinTransformer/Swin-Transformer-Object-Detection). 

**First of all, due to limited time, I did not experiment on the COCO dataset. All results are built on my private dataset, which cannot be shared.** The composition of my dataset is not complicated, with only one class of targets, \~ 1w training images and about \~ 1.5k test images. 

I used the official Swin's pretrained model (https://github.com/microsoft/Swin-Transformer) and the detection version Swin's pretrained model (https://github.com/SwinTransformer/Swin-Transformer-Object-Detection) for experiments. **My experimental results show that using COCO pre-training model works better than using ImageNet pre-training model**. The pretrained model type can be set directly in the configuration file.

For YOLOX with Swin backbone, I set the depth and width factor of PANet neck part with fixed 1.00,  for example, ` self.depth = 1.00 self.width = 1.00` in config file. I simply replaced the backbone part with Swin-T/S/B.


## Usage
For example,
```python
python tools/train.py -f exps/default/yolox_swinB_coco_.py -d 8 -b 64 --fp16 --cache
```
## Results (My private dataset, not COCO !)

#### Standard Models.

|Model      |size   |mAP<sup>test<br>0.5:0.95 |
| ------    | :---: | :---:                   |
|YOLOX-m    |640    |77.04                     |
|YOLOX-l    |640    |72.51                     |
|YOLOX-x    |640    |**78.07**                     |

#### ImageNet Pretrained Models.
`To use ImageNet pre-training, please download the pre-trained model from the [website](https://github.com/microsoft/Swin-Transformer) and place it in the ./pretrained directory.`
|Backbone   |size   |mAP<sup>test<br>0.5:0.95  | pretrained model|
| ------    | :---: | :---:                    | :---:                   |
|swin-base  |320    |72.85                     |swin_base_patch4_window7_224_22k.pth |

#### COCO Pretrained Models.
`To use COCO pre-training, please download the pre-trained model from the [website](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection) and place it in the ./pretrained directory.`
|Backbone   |size   |mAP<sup>test<br>0.5:0.95 |  pretrained model|
| ------    | :---: | :---:                   | :---:                   |
|swin-small |320    |73.72                    | mask_rcnn_swin_tiny_patch4_window7_3x|
|swin-base  |320    |75.06                    | cascade_mask_rcnn_swin_base_patch4_window7_3x|
|swin-tiny  |640    |76.10                    | mask_rcnn_swin_tiny_patch4_window7_3x|
|swin-small |640    |76.81                    | mask_rcnn_swin_tiny_patch4_window7_3x|
|swin-base  |640    |**77.25**                    | cascade_mask_rcnn_swin_base_patch4_window7_3x|

## Some Records

- the curve of yolox_m with size 640

- the curve of yolox with swin-S backbone & size 320

- the curve of yolox with swin-S backbone & size 320





