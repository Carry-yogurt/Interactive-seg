import torch
from functools import partial
from easydict import EasyDict as edict
from albumentations import *
import cv2
train_augmentator = Compose([
    # UniformRandomResize(scale_range=(0.75, 1.40)),
    HorizontalFlip(),
    # PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
    # RandomCrop(*crop_size),
    RandomBrightnessContrast(brightness_limit=(-0.25, 0.25), contrast_limit=(-0.15, 0.4), p=0.75),
    RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.75)
], p=1.0)

img = cv2.imread("./uint/data/COCO_LVIS/images/000000000034.jpg")
cv2.imshow("img",img)


