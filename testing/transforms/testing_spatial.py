import os, sys
sys.path.insert(0, os.getcwd())
import numpy as np
import torch
from horizonms.transforms import (ShearX, ShearY, TranslateX, TranslateY, \
           CropX, CropY, Fliplr, Flipud, Rotate, \
           Resize, ResizeWidth, RandomResizedCrop, RandomCrop, \
           ImagePadding, ImageHeightPaddingOrCrop, \
           RandomShearX, RandomShearY, RandomTranslateX, RandomTranslateY, \
           RandomCropX, RandomCropY, RandomFliplr, RandomFlipud, RandomRotate,
           )


if __name__ == "__main__":
    image = 255 * torch.rand(128, 128, 3)
    print(image.min(), image.max())
    shear_degree = (0, 30)
    op = RandomShearX(0.5, shear_degree)
    for k in range(4):
        print(f"---{k}---")
        print(op)
        print(op.shear_degree)
        image1 = op(image)
        print(op.randomness)
