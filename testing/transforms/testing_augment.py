import os, sys
sys.path.insert(0, os.getcwd())

import torch
import warnings
warnings.filterwarnings("ignore")
import copy
import cv2
import numpy as np
from utils import urandom_int
from torchvision.transforms import InterpolationMode
from horizonms.transforms.base import TargetStructure
from horizonms import transforms as T


def random_target_generator(img, target_type, num_head=1):
    value_pt = None
    value_cv = None
    if target_type == 'bboxes':
        bboxes = []
        for i in range(num_head):
            (h, w, _) = img.shape  # 0-axis为rows,所以是h,1-axis是cols,所以是w
            x1 = urandom_int(0, w - 1)
            y1 = urandom_int(0, h - 1)
            x2 = urandom_int(x1 + 1, w)
            y2 = urandom_int(y1 + 1, h)
            bboxes.append([x1, y1, x2, y2])
        value_pt = torch.tensor(bboxes).float()
        value_cv = np.vstack(bboxes)
    elif target_type == 'points':
        points = []
        for i in range(num_head):
            (h, w, _) = img.shape
            x = urandom_int(0, w - 1)
            y = urandom_int(0, h - 1)
            points.append([x, y])
        value_pt = torch.tensor(points).float()
        value_cv = np.vstack(points)
    elif target_type == 'masks':
        masks_pt = []
        masks_cv = []
        for i in range(num_head):
            (h, w, _) = img.shape
            x1 = urandom_int(0, w - 1)
            y1 = urandom_int(0, h - 1)
            x2 = urandom_int(x1 + 1, w)
            y2 = urandom_int(y1 + 1, h)
            mask_pt = torch.zeros((h, w)).float()
            mask_pt[y1:y2, x1:x2] = torch.ones((y2 - y1, x2 - x1)).float()
            mask_cv = torch.zeros((h, w)).float()
            mask_cv[y1:y2, x1:x2] = torch.ones((y2 - y1, x2 - x1)).float()
            masks_pt.append(mask_pt)
            masks_cv.append(mask_cv[..., None])
        value_pt = torch.stack(masks_pt, 0)
        value_cv = np.concatenate(masks_cv, axis=-1)
    elif target_type == 'labels':
        labels_pt = []
        labels_cv = []
        for i in range(num_head):
            label = urandom_int(0, 3)
            labels_pt.append(torch.tensor(label))
            labels_cv.append(label)
        value_pt = labels_pt
        value_cv = labels_cv
    target_pt = dict(labels=TargetStructure(type=target_type, value=value_pt))
    target_cv = dict(labels=TargetStructure(type=target_type, value=value_cv))
    return target_pt, target_cv


# custom ta不支持random系列的op
customtrivalaugment =dict(
    augment_operators=[
        # image.py
        # dict(name='Uint8ToFloat'),
        # dict(name='Identity'),
        # dict(name='Brightness', param_range=[0.5, 2.0]),
        # dict(name='Contrast', param_range=[0.5, 2.0]),
        # dict(name='Saturation', param_range=[0.5, 2.0]),
        # dict(name='Hue', param_range=[-0.1, 0.1]),
        # dict(name='Sharpness', param_range=[0.5, 2.0]),
        # dict(name='Posterize', param_range=[2, 8]),
        # dict(name='Solarize', param_range=[1.0, 0.85]),
        # dict(name='AutoContrast'),
        # dict(name='Equalize'),
        # dict(name='Invert')
        # dict(name='GaussianBlur', param_range=[0.1, 1.2], kernel_size=7),
        # dict(name='GaussianNoise', param_range=[-0.1, 0.1], mean=0.0),
        # dict(name='Lighting',
        #      param_range=[0, 1],
        #      eigval=np.array([0.5, 0.0188, 0.0045]),
        #      eigvec=np.array([[-0.5675, 0.7192, 0.4009], [-0.5808, -0.0045, -0.8140], [-0.5836, -0.6948, 0.4203]]))

        # image_cv.py
        # dict(name='CVUint8ToFloat')
        # dict(name='CVIdentity')
        # dict(name='CVBrightness', param_range=[0.5, 2.0])
        # dict(name='CVContrast', param_range=[0.5, 2.0])
        # dict(name='CVSaturation', param_range=[0.5, 2.0])
        # dict(name='CVHue', param_range=[-0.1, 0.1])
        # dict(name='CVSharpness', param_range=[1, 10])
        # dict(name='CVPosterize', param_range=[1, 1])
        # dict(name='CVSolarize', param_range=[1.0, 0.85])
        # dict(name='CVAutoContrast')
        # dict(name='CVEqualize', )
        # dict(name='CVInvert')
        # dict(name='CVGaussianBlur', param_range=[0.1, 1.2], kernel_size=7)
        # dict(name='CVGaussianNoise', param_range=[-0.1, 0.1], mean=0.0)
        # dict(name='CVLighting',
        #      param_range=[0, 1],
        #      eigval=np.array([0.5, 0.0188, 0.0045]),
        #      eigvec=np.array([[-0.5675, 0.7192, 0.4009], [-0.5808, -0.0045, -0.8140], [-0.5836, -0.6948, 0.4203]]))

        # spatial.py
        # dict(name='ShearX', param_range=[-30, 30])
        # dict(name='ShearY', param_range=[-30, 30])
        # dict(name='TranslateX', param_range=[0, 1])
        # dict(name='TranslateY', param_range=[0, 1])
        # dict(name='CropX', param_range=[0, 1])
        # dict(name='CropY', param_range=[0, 1])
        # dict(name='Fliplr')
        # dict(name='Flipud')
        # dict(name='Rotate', param_range=[-60, 60])
        # dict(name='Resize', param_range=[100, 300])  # 没有办法适配param range生成fp，函数要求int
        # dict(name='ResizeWidth', param_range=[100, 300])  # 没有办法适配param range生成fp，函数要求int
        # dict(name='Scale', param_range=[0.8, 1.2])
        # dict(name='ImagePadding')

        # spatial_cv.py
        # dict(name='CVShearX', param_range=[-30, 30])
        # dict(name='CVShearY', param_range=[-30, 30])
        # dict(name='CVTranslateX', param_range=[0, 1])
        # dict(name='CVTranslateY', param_range=[0, 1])
        # dict(name='CVCropX', param_range=[0, 1]),
        # dict(name='CVCropY', param_range=[0, 1]),
        # dict(name='CVFliplr'),
        # dict(name='CVFlipud')
        # dict(name='CVRotate', param_range=[-60, 60]),
        # dict(name='CVResize', param_range=[100, 300])  # 没有办法适配param range生成fp，函数要求int
        # dict(name='CVResizeWidth', param_range=[100, 300])  # 没有办法适配param range生成fp，函数要求int
        # dict(name='CVScale', param_range=[0.8, 1.2])
        # dict(name='CVImagePadding')
    ],
    num_magnitude_bins=32
)


# HorizonmsTrivialAugment也是不支持random系列op
horizonmstrivalaugment = dict(
    augment_operators=[
        # image.py
        # dict(name='Uint8ToFloat'),
        # dict(name='Identity'),
        # dict(name='Normalizer', mode='zscore'),
        # dict(name='Normalizer', mode='zero-one'),
        # dict(name='Normalizer', mode='negative-positive-one'),
        # dict(name='Normalizer', mode='customize',  shift=[1], scale=[0.9]),
        # dict(name='Normalizer', mode='zscore', image_base=False, epsilon=1e-6),
        # dict(name='Brightness', brightness_factor=[0.5, 2.0]),
        # dict(name='Brightness', brightness_factor=0.6),
        # dict(name='Brightness', brightness_factor=(0, 1)),
        # dict(name='Contrast', contrast_factor=0.7),
        # dict(name='Saturation', saturation_factor=(0, 1))
        # dict(name='Hue', hue_factor=[-0.3, 0.4])
        # dict(name='Sharpness', sharpness_factor=0.9)
        # dict(name='Posterize', posterize_bins=6),
        # dict(name='Solarize', solarize_threshold=(0, 1))
        # dict(name='AutoContrast')
        # dict(name='Equalize')
        # dict(name='Invert')
        # dict(name='GaussianBlur', sigma=0.1, kernel_size=(3, 3))
        # dict(name='GaussianBlur', sigma=(0, 3), kernel_size=(5, 7))
        # dict(name='GaussianNoise', std=1.0, mean=0.0)
        # dict(name='GaussianNoise', std=(0, 2), mean=[2.3, 3.7])
        # dict(name='Lighting',
        #      alphastd=(0,1),
        #      eigval=np.array([0.5, 0.0188, 0.0045]),
        #      eigvec=np.array([[-0.5675, 0.7192, 0.4009], [-0.5808, -0.0045, -0.8140], [-0.5836, -0.6948, 0.4203]]))

        # image_cv.py
        # dict(name='CVUint8ToFloat')
        # dict(name='CVIdentity')
        # dict(name='CVCvtColor', code='cv2.COLOR_RGB2BGR')
        # dict(name='CVNormalizer', mode='zscore'),
        # dict(name='CVNormalizer', mode='zero-one', epsilon=2e-10),
        # dict(name='CVNormalizer', mode='negative-positive-one', shift=0.1, image_base=False),
        # dict(name='CVNormalizer', mode='customize', shift=[0.1], scale=[0.7]),
        # dict(name='CVBrightness', brightness_factor=(0, 1))
        # dict(name='CVContrast', contrast_factor=(0, 1))
        # dict(name='CVSaturation', saturation_factor=[0.5, 2.0])
        # dict(name='CVHue', hue_factor=[-0.1, 0.1])
        # dict(name='CVSharpness', sharpness_factor=(1, 10))
        # dict(name='CVPosterize', posterize_bins=(2, 7))
        # dict(name='CVSolarize', solarize_threshold=(1.0, 0.85))
        # dict(name='CVAutoContrast')
        # dict(name='CVEqualize')
        # dict(name='CVInvert')
        # dict(name='CVGaussianBlur', sigma=(0.1, 1.2), kernel_size=[3, 5, 9])
        # dict(name='CVGaussianNoise', std=(-0.1, 0.1), mean=(-1.0, 1.0))
        # dict(name='CVLighting',
        #      alphastd=[0, 1],
        #      eigval=np.array([0.5, 0.0188, 0.0045]),
        #      eigvec=np.array([[-0.5675, 0.7192, 0.4009], [-0.5808, -0.0045, -0.8140], [-0.5836, -0.6948, 0.4203]]))

        # spatial.py
        # dict(name='ShearX', shear_degree=[-30, 30])
        # dict(name='ShearY', shear_degree=[-30, 30])
        # dict(name='TranslateX', translate_ratio=(0, 1))
        # dict(name='TranslateY', translate_ratio=0.9)
        # dict(name='CropX', crop_ratio=(0, 1))
        # dict(name='CropY', crop_ratio=0.4)
        # dict(name='Fliplr')
        # dict(name='Flipud')
        # dict(name='Rotate', rotate_degree=(-60, 60))
        # dict(name='Resize', size=[100, 300])
        # dict(name='Resize', size=500, interpolation=InterpolationMode.NEAREST)
        # dict(name='Scale', scale_range=(0.8, 1.2))
        # dict(name='Scale', scale_range=(0.8, 1.2), scale_same=True, interpolation=InterpolationMode.NEAREST)
        # dict(name='ImagePadding')
        # dict(name='ImageHeightPaddingOrCrop')

        # spatial_cv.py
        # dict(name='CVShearX', shear_degree=(-30, 30))
        # dict(name='CVShearY', shear_degree=[-30, 30])
        # dict(name='CVTranslateX', translate_ratio=(0, 1))
        # dict(name='CVTranslateY', translate_ratio=[0, 1])
        # dict(name='CVCropX', crop_ratio=(0, 1)),
        # dict(name='CVCropY', crop_ratio=[0, 1]),
        # dict(name='CVFliplr'),
        # dict(name='CVFlipud')
        # dict(name='CVRotate', rotate_degree=[-60, 60]),
        # dict(name='CVResize', size=[100, 300])
        # dict(name='CVResize', size=(700, 700), interpolation=cv2.INTER_CUBIC)
        # dict(name='CVResizeWidth', width=700)
        # dict(name='CVScale', scale_range=[0.8, 1.2], scale_same=True, interpolation='cv2.INTER_CUBIC')
        # dict(name='CVImagePadding')
    ]
)


# SequentialAugment相当于在HorizonmsTrivialAugment基础上支持了random系列的op，这里只测random系列的
sequentialaugment = dict(
    augment_operators=[
        # image.py
        # dict(name='RandomBrightness', prob=1, brightness_factor=(0, 1)),
        # dict(name='RandomContrast', prob=1, contrast_factor=[0, 2]),
        # dict(name='RandomSaturation', prob=1, saturation_factor=0.6),
        # dict(name='RandomHue', prob=1, hue_factor=(-0.5, 0.5)),
        # dict(name='RandomSharpness', prob=1, sharpness_factor=(0, 2)),
        # dict(name='RandomPosterize', prob=1, posterize_bins=[0, 0.7, 1]),
        # dict(name='RandomPosterize', prob=1, posterize_bins=2.8),
        # dict(name='RandomSolarize', prob=1, solarize_threshold=(0, 1))
        # dict(name='RandomAutoContrast', prob=1),
        # dict(name='RandomEqualize', prob=1),
        # dict(name='RandomInvert', prob=1),
        # dict(name='RandomGaussianBlur', prob=1, sigma=(0, 5), kernel_size=(3,5)),
        # dict(name='RandomGaussianNoise', prob=1, mean=1, std=1)
        # dict(name='RandomLighting', prob=1),

        # image_cv.py
        # dict(name='CVRandomBrightness', prob=1, brightness_factor=(0, 0.4))
        # dict(name='CVRandomContrast', prob=1, contrast_factor=(0, 3))
        # dict(name='CVRandomSaturation', prob=1, saturation_factor=[0.1, 0.9, 1.4]),
        # dict(name='CVRandomHue', prob=1, hue_factor=0.1),
        # dict(name='CVRandomSharpness', prob=1, sharpness_factor=(0, 1))
        # dict(name='CVRandomPosterize', prob=1, posterize_bins=[0, 1, 8])
        # dict(name='CVRandomSolarize', prob=1, solarize_threshold=(0, 1)),
        # dict(name='CVRandomAutoContrast', prob=1),
        # dict(name='CVRandomEqualize', prob=1),
        # dict(name='CVRandomInvert', prob=1),
        # dict(name='CVRandomGaussianBlur', prob=1, sigma=(0, 3), kernel_size=[3, 9, 11]),
        # dict(name='CVRandomGaussianNoise', prob=1, mean=(-3, 3), std=(-1, -1)),
        # dict(name='CVRandomLighting', prob=1),

        # spatial.py
        # dict(name='RandomResizedCrop', size=(500, 800))
        # dict(name='RandomCrop', crop_size=300, prob=0.5, mask_type='masks', obj_labels=1)  # 会报错未解决
        # dict(name='RandomShearX', prob=1, shear_degree=30)
        # dict(name='RandomShearY', prob=1, shear_degree=(-30, 30))
        # dict(name='RandomTranslateX', prob=1, translate_ratio=(0, 1))
        # dict(name='RandomTranslateY', prob=1, translate_ratio=0.6)
        # dict(name='RandomCropX', prob=1, crop_ratio=[0, 0.3, 0.8]),
        # dict(name='RandomCropY', prob=1, crop_ratio=(0, 1)),
        # dict(name='RandomFliplr', prob=1),
        # dict(name='RandomFlipud', prob=1),
        # dict(name='RandomRotate', prob=1, rotate_degree=[-30, 30]),
        # dict(name='RandomScale', prob=1, scale_range=(0, 2), scale_width=True, scale_height=True,
        #      interpolation='InterpolationMode.NEAREST')

        # spatial_cv.py
        dict(name='CVRandomResizedCrop', size=(300, 500), interpolation='cv2.INTER_NEAREST')
        # dict(name='CVRandomCrop', prob=1, crop_ratio=[0, 1, 2])
        # dict(name='CVRandomScale', prob=1, scale_range=(0, 1), scale_width=True, scale_height=True,
        #      interpolation='cv2.INTER_NEAREST')
        # dict(name='CVRandomShift', prob=1, shift_limit=(0, 1))
        # dict(name='CVRandomShearX', prob=1, shear_degree=(-180, 180)),
        # dict(name='CVRandomShearY', prob=1, shear_degree=(-180, 180)),
        # dict(name='CVRandomTranslateX', prob=1, translate_ratio=(0, 1)),
        # dict(name='CVRandomTranslateY', prob=1, translate_ratio=(0, 1)),
        # dict(name='CVRandomCropX', prob=1, crop_ratio=(0, 1)),
        # dict(name='CVRandomCropY', prob=1, crop_ratio=(0, 1)),
        # dict(name='CVRandomFliplr', prob=1)
        # dict(name='CVRandomFlipud', prob=1),
        # dict(name='CVRandomRotate', prob=1, rotate_degree=[-30, -10, 10, 30]),
    ]
)


if __name__ == "__main__":

    image_path = '../../data/test_color_img'
    image_files = [os.path.join(image_path, i) for i in os.listdir(image_path)][:1]

    # custom_ta = T.CustomizedTrivialAugment(customtrivalaugment['augment_operators'],
    #                                        customtrivalaugment['num_magnitude_bins'])
    # ta = custom_ta

    # horizonms_ta = T.HorizonmsTrivialAugment(horizonmstrivalaugment['augment_operators'])
    # ta = horizonms_ta

    sequential_ta = T.SequentialAugment(sequentialaugment['augment_operators'])
    ta = sequential_ta

    to_tensor_fp = T.ToTensor('float')
    to_tensor_int = T.ToTensor('uint8')

    for image_file in image_files:
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_copy = image.copy()
        # 分别检查float和int类型的image

        # 对于image.py与 image_cv.py，不用生成不同类型target
        # 对于pt的op，需要转为tensor，执行下面的
        # image_tensor_fp = to_tensor_fp(image)
        # image_tensor_int = to_tensor_int(image)
        # ta_fp = ta(image_tensor_fp)
        # ta_int = ta(image_tensor_int)
        # 对于cv的op，不需要转化为tensor，执行下面的
        # ta_fp = ta((image/255.0).astype(np.float32))
        # ta_int = ta(image)

        # 对于spatial.py 与spatial_cv.py生成5个label传入
        for target_type in ['masks', 'labels', None, 'bboxes', 'points']:
        # for target_type in ['points']:

            target_pt1, target_cv1 = random_target_generator(image_copy, target_type)
            target_pt2, target_cv2 = random_target_generator(image_copy, target_type)

            # 对于pt的op，需要转为tensor，执行下面的
            # image_tensor_fp = to_tensor_fp(image)
            # image_tensor_int = to_tensor_int(image)
            # ta_fp = ta(image_tensor_fp, target_pt1)
            # ta_int = ta(image_tensor_int, target_pt2)

            # 对于cv的op，不需要转化为tensor，执行下面的
            ta_fp = ta((image/255.0).astype(np.float32), target_cv1)
            ta_int = ta(image, target_cv2)

