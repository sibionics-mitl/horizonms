import math
from typing import Tuple, List, Optional, Sequence
import torch
from torch import Tensor
from torchvision.transforms import functional as F, InterpolationMode
import warnings
import random
from abc import ABC, abstractmethod
from .utils import setup_size, _input_check_value_range_set, _input_get_value_range_set
from ..builder import TRANSFORMS


__all__ = ("SpatialBase", "ShearX", "ShearY", "TranslateX", "TranslateY",
           "CropX", "CropY", "Fliplr", "Flipud", "Rotate", "Scale",
           "Resize", "ResizeWidth", "RandomResizedCrop", "RandomCrop",
           "ImagePadding", "ImageHeightPaddingOrCrop",
           "RandomShearX", "RandomShearY", "RandomTranslateX", "RandomTranslateY",
           "RandomCropX", "RandomCropY", "RandomFliplr", "RandomFlipud", "RandomRotate", "RandomScale")


class SpatialBase(ABC, torch.nn.Module):

    @abstractmethod
    def calculate_image(self, image):
        pass

    @abstractmethod
    def calculate_target(self, target):
        pass

    def forward(self, image, target=None):
        image = self.calculate_image(image)
        if target is None:
            return image
        else:
            target = self.calculate_target(target)
            return image, target


def _format_fill(fill, image):
    if isinstance(fill, (int, float)):
        fill = [float(fill)] * F.get_image_num_channels(image)
    elif fill is not None:
        fill = [float(f) for f in fill]
    return fill


def _shear_x_image(image, shear_degree, fill):
    fill = _format_fill(fill, image)
    image = F.affine(
        image,
        angle=0.0,
        translate=[0, 0],
        scale=1.0,
        shear=[shear_degree, 0.0],
        # interpolation=interpolation, ## cv版无此参数,为了统一,这里使用函数内置默认值
        fill=fill,
    )
    return image


def _shear_x_target(target, shear_degree):
    for key, value in target.items():
        if value.type not in ['masks', 'bboxes', 'points']:
            continue
        if value.type == 'masks':
            if value.islist:
                shear = [F.affine(v, angle=0.0, translate=[0, 0], scale=1.0,
                                  shear=[shear_degree, 0.0])
                         for v in value.value]
            else:
                shear = F.affine(value.value, angle=0.0, translate=[0, 0], scale=1.0,
                                 shear=[shear_degree, 0.0])
        if value.type in ['bboxes', 'points']:
            raise ValueError(f"shear does not support type={value.type}.")
        value.value = shear
        target[key] = value
    return target


@TRANSFORMS.register_module()
class ShearX(SpatialBase):
    """Shear images along x-axis (width).

    Args:
        shear_degree (float  | tuple[float, float] |  list[float, float] ):
                float: Shear angle in degree between -180 and 180, clockwise direction.
                        When shear_degree is not None, it is used for image shearing.
                tuple[float, float]: Range of shear angle in degree.
                list[float, ... , float] : list of the ratio of shear angle to be randomly chosen.
        fill (sequence or number, optional): Pixel fill value for the area outside the
                transformed image. If given a number, the value is used for all bands respectively.
    """

    def __init__(self, shear_degree: float,
                 fill: Optional[List[float]] = None):
        super().__init__()
        _input_check_value_range_set(shear_degree)
        self.shear_degree = shear_degree
        self.fill = fill

    def calculate_image(self, image):
        self.shear_degree_sel = _input_get_value_range_set(self.shear_degree)
        return _shear_x_image(image, self.shear_degree_sel, self.fill)

    def calculate_target(self, target):
        return _shear_x_target(target, self.shear_degree_sel)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(shear_degree={self.shear_degree}, '
        repr_str += f'fill={self.fill})'
        return repr_str


@TRANSFORMS.register_module()
class RandomShearX(SpatialBase):
    """Shear images along x-axis (width).

    Args:
        prob (float):probability of the image being sheared.
        shear_degree (float  | tuple[float, float] |  list[float, float] ):
                float: Shear angle in degree between -180 and 180, clockwise direction.
                        When shear_degree is not None, it is used for image shearing.
                tuple[float, float]: Range of shear angle in degree.
                list[float, ... , float] : list of the ratio of shear angle to be randomly chosen.
        fill (sequence or number, optional): Pixel fill value for the area outside the
                transformed image. If given a number, the value is used for all bands respectively.
    """

    def __init__(self, prob: float, shear_degree: Tuple[float] = None,
                 fill: Optional[List[float]] = None):
        super().__init__()
        _input_check_value_range_set(shear_degree)
        self.prob = prob
        self.shear_degree = shear_degree
        self.fill = fill        

    def calculate_image(self, image):
        self.randomness = False
        if random.random() < self.prob:
            self.randomness = True
            self.shear_degree_sel = _input_get_value_range_set(self.shear_degree)
            image = _shear_x_image(image, self.shear_degree_sel, self.fill)
        return image

    def calculate_target(self, target):
        if self.randomness:
            target = _shear_x_target(target, self.shear_degree_sel)
        return target

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'shear_degree={self.shear_degree}, '
        repr_str += f'fill={self.fill})'
        return repr_str


def _shear_y_image(image, shear_degree, fill):
    fill = _format_fill(fill, image)
    image = F.affine(
        image,
        angle=0.0,
        translate=[0, 0],
        scale=1.0,
        shear=[0.0, shear_degree],
        # interpolation=interpolation, ## cv版无此参数,为了统一,这里使用函数内置默认值
        fill=fill,
    )
    return image


def _shear_y_target(target, shear_degree):
    for key, value in target.items():
        if value.type not in ['masks', 'bboxes', 'points']:
            continue
        if value.type == 'masks':
            if value.islist:
                shear = [F.affine(v, angle=0.0, translate=[0, 0], scale=1.0,
                                  shear=[0.0, shear_degree])
                         for v in value.value]
            else:
                shear = F.affine(value.value, angle=0.0, translate=[0, 0], scale=1.0,
                                 shear=[0.0, shear_degree])
        if value.type in ['bboxes', 'points']:
            raise ValueError(f"shear does not support type={value.type}.")
        value.value = shear
        target[key] = value
    return target


@TRANSFORMS.register_module()
class ShearY(SpatialBase):
    """Shear images along y-axis (height).

    Args:
        shear_degree (float  | tuple[float, float] |  list[float, float] ):
                float: Shear angle in degree between -180 and 180, clockwise direction.
                        When shear_degree is not None, it is used for image shearing.
                tuple[float, float]: Range of shear angle in degree.
                list[float, ... , float] : list of the ratio of shear angle to be randomly chosen.
        fill (sequence or number, optional): Pixel fill value for the area outside the
                transformed image. If given a number, the value is used for all bands respectively.
    """

    def __init__(self, shear_degree,
                 fill: Optional[List[float]] = None):
        super().__init__()
        _input_check_value_range_set(shear_degree)
        self.shear_degree = shear_degree
        self.fill = fill

    def calculate_image(self, image):
        self.shear_degree_sel = _input_get_value_range_set(self.shear_degree)
        return _shear_y_image(image, self.shear_degree_sel, self.fill)

    def calculate_target(self, target):
        return _shear_y_target(target, self.shear_degree_sel)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(shear_degree={self.shear_degree}, '
        repr_str += f'fill={self.fill})'
        return repr_str


@TRANSFORMS.register_module()
class RandomShearY(SpatialBase):
    """Shear images along y-axis (height).

    Args:
        shear_degree (float  | tuple[float, float] |  list[float, float] ):
                float: Shear angle in degree between -180 and 180, clockwise direction.
                        When shear_degree is not None, it is used for image shearing.
                tuple[float, float]: Range of shear angle in degree.
                list[float, ... , float] : list of the ratio of shear angle to be randomly chosen.
        fill (sequence or number, optional): Pixel fill value for the area outside the
                transformed image. If given a number, the value is used for all bands respectively.
    """

    def __init__(self, prob, shear_degree: Tuple[float] = None,
                 fill: Optional[List[float]] = None):
        super().__init__()
        _input_check_value_range_set(shear_degree)
        self.prob = prob
        self.shear_degree = shear_degree
        self.fill = fill

    def calculate_image(self, image):
        self.randomness = False
        if random.random() < self.prob:
            self.randomness = True
            self.shear_degree_sel = _input_get_value_range_set(self.shear_degree)
            image = _shear_y_image(image, self.shear_degree_sel, self.fill)
        return image

    def calculate_target(self, target):
        if self.randomness:
            target = _shear_y_target(target, self.shear_degree_sel)
        return target

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'shear_degree={self.shear_degree}, '
        repr_str += f'fill={self.fill})'
        return repr_str


def _translate_x_image(image, translate_ratio, fill):
    fill = _format_fill(fill, image)
    image_width = image.shape[-1]
    magnitude = int(translate_ratio * image_width)
    image = F.affine(
        image,
        angle=0.0,
        translate=[magnitude, 0],
        scale=1.0,
        # interpolation=interpolation, ## cv版无此参数,为了统一,这里使用函数内置默认值
        shear=[0.0, 0.0],
        fill=fill,
    )
    return image


def _translate_x_target(target, translate_ratio, image_width):
    magnitude = int(translate_ratio * image_width)
    for key, value in target.items():
        if value.type not in ['masks', 'bboxes', 'points']:
            continue
        if value.type == 'masks':
            if value.islist:
                translate = [F.affine(v, angle=0.0, translate=[magnitude, 0],
                                      scale=1.0, shear=[0.0, 0.0])
                             for v in value.value]
            else:
                translate = F.affine(value.value, angle=0.0, translate=[magnitude, 0],
                                     scale=1.0, shear=[0.0, 0.0])
        if value.type == 'bboxes':
            translate = value.value
            translate[:, [0, 2]] = torch.clamp(translate[:, [0, 2]] + magnitude, 0, image_width - 1)
            flag = (translate[:, 2] - translate[:, 0]) > 1
            translate = translate[flag, :]
        if value.type == 'points':
            translate = value.value
            translate[:, 0] = torch.clamp(translate[:, 0] + magnitude, 0, image_width - 1)
            flag = (translate[:, 0] > 0) & (translate[:, 0] < image_width - 1)
            translate = translate[flag, :]
        value.value = translate
        target[key] = value
    return  target


@TRANSFORMS.register_module()
class TranslateX(SpatialBase):
    """Translate images along x-axis (width), images size = (h, w).

    Args:
        translate_ratio (float  | tuple[float, float] |  list[float, float] ):
                float: Ratio of translation in the range of [0, 1].
                tuple[float, float]: Range of the ratio of translation.
                list[float, ... , float] : list of the ratio of translation to be randomly chosen.
        fill (sequence or number, optional): Pixel fill value for the area outside the
                transformed image. If given a number, the value is used for all bands respectively.
        When translate_ratio is positve, the image moves to the right.
        When translate_ratio is negative, the image moves to the left.
    """

    def __init__(self, translate_ratio: float = 0.0,
                 fill: Optional[List[float]] = None):
        super().__init__()
        _input_check_value_range_set(translate_ratio)
        self.translate_ratio = translate_ratio
        self.fill = fill

    def calculate_image(self, image):
        self.image_width = image.shape[-1]
        self.translate_ratio_sel = _input_get_value_range_set(self.translate_ratio)
        return _translate_x_image(image, self.translate_ratio_sel, self.fill)

    def calculate_target(self, target):
        return _translate_x_target(target, self.translate_ratio_sel, self.image_width)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(translate_ratio={self.translate_ratio}, '
        repr_str += f'fill={self.fill})'
        return repr_str


@TRANSFORMS.register_module()
class RandomTranslateX(SpatialBase):
    """Translate images along x-axis (width), images size = (h, w).

    Args:
        translate_ratio (float  | tuple[float, float] |  list[float, float] ):
                float: Ratio of translation in the range of [0, 1].
                tuple[float, float]: Range of the ratio of translation.
                list[float, ... , float] : list of the ratio of translation to be randomly chosen.
        fill (sequence or number, optional): Pixel fill value for the area outside the
                transformed image. If given a number, the value is used for all bands respectively.
        When translate_ratio is positve, the image moves to the right.
        When translate_ratio is negative, the image moves to the left.
    """

    def __init__(self, prob: float = 0.0, translate_ratio=None,
                 fill: Optional[List[float]] = None):
        super().__init__()
        _input_check_value_range_set(translate_ratio)
        self.prob = prob        
        self.translate_ratio = translate_ratio
        self.fill = fill

    def calculate_image(self, image):
        self.randomness = False
        if random.random() < self.prob:
            self.randomness = True
            self.image_width = image.shape[-1]
            self.translate_ratio_sel = _input_get_value_range_set(self.translate_ratio)
            image = _translate_x_image(image, self.translate_ratio_sel, self.fill)
        return image

    def calculate_target(self, target):
        if self.randomness:
            target = _translate_x_target(target, self.translate_ratio_sel, self.image_width)
        return target

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'translate_ratio={self.translate_ratio}, '
        repr_str += f'fill={self.fill})'
        return repr_str


def _translate_y_image(image, translate_ratio, fill):
    fill = _format_fill(fill, image)
    image_height = image.shape[-2]
    magnitude = int(translate_ratio * image_height)
    image = F.affine(
        image,
        angle=0.0,
        translate=[0, magnitude],
        scale=1.0,
        # interpolation=interpolation, ## cv版无此参数,为了统一,这里使用函数内置默认值
        shear=[0.0, 0.0],
        fill=fill,
    )
    return image


def _translate_y_target(target, translate_ratio, image_height):
    magnitude = int(translate_ratio * image_height)
    for key, value in target.items():
        if value.type not in ['masks', 'bboxes', 'points']:
            continue
        if value.type == 'masks':
            if value.islist:
                translate = [F.affine(v, angle=0.0, translate=[0, magnitude],
                                      scale=1.0, shear=[0.0, 0.0])
                             for v in value.value]
            else:
                translate = F.affine(value.value, angle=0.0, translate=[0, magnitude],
                                     scale=1.0, shear=[0.0, 0.0])
        if value.type == 'bboxes':
            translate = value.value
            translate[:, [1, 3]] = torch.clamp(translate[:, [1, 3]] + magnitude, 0, image_height - 1)
            flag = (translate[:, 3] - translate[:, 1]) > 1
            translate = translate[flag, :]
        if value.type == 'points':
            translate = value.value
            translate[:, 1] = torch.clamp(translate[:, 1] + magnitude, 0, image_height - 1)
            flag = (translate[:, 1] > 0) & (translate[:, 1] < image_height - 1)
            translate = translate[flag, :]
        value.value = translate
        target[key] = value
    return target


@TRANSFORMS.register_module()
class TranslateY(SpatialBase):
    """Translate images along y-axis (height), images size = (h, w).

    Args:
        translate_ratio  (float  | tuple[float, float] |  list[float, float] ):
                float: Ratio of translation in the range of [0, 1].
                tuple[float, float]: Range of the ratio of translation.
                list[float, ... , float] : list of the ratio of translation to be randomly chosen.
        fill (sequence or number, optional) - Pixel fill value for the area outside the
                transformed image. If given a number, the value is used for all bands respectively.
        When translate_ratio is positve, the image moves down.
        When translate_ratio is negative, the image moves up.
    """

    def __init__(self, translate_ratio,
                 fill: Optional[List[float]] = None):
        super().__init__()

        _input_check_value_range_set(translate_ratio)
        self.translate_ratio = translate_ratio
        self.fill = fill

    def calculate_image(self, image):
        self.image_height = image.shape[-2]
        self.translate_ratio_sel = _input_get_value_range_set(self.translate_ratio)
        return _translate_y_image(image, self.translate_ratio_sel, self.fill)

    def calculate_target(self, target):
        return _translate_y_target(target, self.translate_ratio_sel, self.image_height)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(translate_ratio={self.translate_ratio}, '
        repr_str += f'fill={self.fill})'
        return repr_str


@TRANSFORMS.register_module()
class RandomTranslateY(SpatialBase):
    """Translate images along y-axis (height), images size = (h, w).

    Args:
        translate_prob (float): Probability of the image being translated.
        translate_ratio  (float  | tuple[float, float] |  list[float, float] ):
                float: Ratio of translation in the range of [0, 1].
                tuple[float, float]: Range of the ratio of translation.
                list[float, ... , float] : list of the ratio of translation to be randomly chosen.
        fill (sequence or number, optional) - Pixel fill value for the area outside the
                transformed image. If given a number, the value is used for all bands respectively.
        When translate_ratio is positve, the image moves down.
        When translate_ratio is negative, the image moves up.
    """

    def __init__(self, prob, translate_ratio=None,
                 fill: Optional[List[float]] = None):
        super().__init__()
        _input_check_value_range_set(translate_ratio)
        self.prob = prob
        self.translate_ratio = translate_ratio
        self.fill = fill

    def calculate_image(self, image):
        self.randomness = False
        if random.random() < self.prob:
            self.randomness = True
            self.image_height = image.shape[-2]
            self.translate_ratio_sel = _input_get_value_range_set(self.translate_ratio)
            image = _translate_y_image(image, self.translate_ratio_sel, self.fill)
        return image

    def calculate_target(self, target):
        if self.randomness:
            target = _translate_y_target(target, self.translate_ratio_sel, self.image_height)
        return target

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'translate_ratio={self.translate_ratio}, '
        repr_str += f'fill={self.fill})'
        return repr_str


def _crop_x_image(image, crop_ratio):
    image_width = image.shape[-1]
    magnitude = int(crop_ratio * image_width)
    if magnitude >= 0:
        image = image[:, :, magnitude:]
    else:
        image = image[:, :, :magnitude]
    return image


def _crop_x_target(target, crop_ratio, image_width):
    magnitude = int(crop_ratio * image_width)
    for key, value in target.items():
        if value.type not in ['masks', 'bboxes', 'points']:
            continue
        if value.type == 'masks':
            if value.islist:
                if magnitude >= 0:
                    crop = [v[:, :, magnitude:] for v in value.value]
                else:
                    crop = [v[:, :, :magnitude] for v in value.value]
            else:
                if magnitude >= 0:
                    crop = value.value[:, :, magnitude:]
                else:
                    crop = value.value[:, :, :magnitude]
        if value.type == 'bboxes':
            crop = value.value
            if magnitude >= 0:
                crop[:, [0, 2]] = torch.clamp(crop[:, [0, 2]] - magnitude, 0, image_width - 1)
            else:
                crop[:, [0, 2]] = torch.clamp(crop[:, [0, 2]], 0, image_width - 1)
            flag = (crop[:, 2] - crop[:, 0]) > 1
            crop = crop[flag, :]
        if value.type == 'points':
            crop = value.value

            if magnitude >= 0:
                crop[:, 0] = torch.clamp(crop[:, 0] - magnitude, 0, image_width - 1)
            else:
                crop[:, 0] = torch.clamp(crop[:, 0], 0, image_width - 1)

            flag = (crop[:, 0] > 0) & (crop[:, 0] < image_width - 1)
            crop = crop[flag, :]

        value.value = crop
        target[key] = value
    return  target


@TRANSFORMS.register_module()
class CropX(SpatialBase):
    """Crop images along x-axis (width), images size = (h, w).

    Args:
        crop_ratio  (float  | tuple[float, float] |  list[float, float] ):
            float: Ratio of cropping in the range of [0, 1].
                   When crop_ratio is not None, it crops an image to (h, crop_ratio*w).
            tuple[float, float]: Range of the ratio of cropping. Default is None.
                    When crop_ratio=[min_ratio, max_ratio] is None, it crops
                    an image to (h, ratio*w), in which ratio is randomly selected from [min_ratio, max_ratio].
            list[float, ... , float] : list of the ratio of cropping to be randomly chosen.
        When crop_ratio is positve, the left portion of the image is cropped out, and the right portion is kept.
        When crop_ratio is negative, the left portion of the image is kept, and the right portion is cropped out.
    """

    def __init__(self, crop_ratio: float):
        super().__init__()
        _input_check_value_range_set(crop_ratio)
        self.crop_ratio = crop_ratio

    def calculate_image(self, image):
        self.image_width = image.shape[-1]
        self.crop_ratio_sel = _input_get_value_range_set(self.crop_ratio)
        return _crop_x_image(image, self.crop_ratio_sel)

    def calculate_target(self, target):
        return _crop_x_target(target, self.crop_ratio_sel, self.image_width)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(crop_ratio={self.crop_ratio})'
        return repr_str


@TRANSFORMS.register_module()
class RandomCropX(SpatialBase):
    """Crop images along x-axis (width), images size = (h, w).

    Args:
        crop_prob (float):Probability of the image being cropped.
        crop_ratio  (float  | tuple[float, float] |  list[float, float] ):
            float: Ratio of cropping in the range of [0, 1].
                    When crop_ratio is not None, it crops an image to (h, crop_ratio*w).
            tuple[float, float]: Range of the ratio of cropping. Default is None.
                    When crop_ratio=[min_ratio, max_ratio] is None, it crops
                    an image to (h, ratio*w), in which ratio is randomly selected from [min_ratio, max_ratio].
            list[float, ... , float] : list of the ratio of cropping to be randomly chosen.
        When crop_ratio is positve, the left portion of the image is cropped out, and the right portion is kept.
        When crop_ratio is negative, the left portion of the image is kept, and the right portion is cropped out.
    """

    def __init__(self, prob: float, crop_ratio=None):
        super().__init__()
        _input_check_value_range_set(crop_ratio)
        self.prob = prob
        self.crop_ratio = crop_ratio

    def calculate_image(self, image):
        self.randomness = False
        if random.random() < self.prob:
            self.randomness = True
            self.image_width = image.shape[-1]
            self.crop_ratio_sel = _input_get_value_range_set(self.crop_ratio)
            image = _crop_x_image(image, self.crop_ratio_sel)
        return image

    def calculate_target(self, target):
        if self.randomness:
            target = _crop_x_target(target, self.crop_ratio_sel, self.image_width)
        return target

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'crop_ratio={self.crop_ratio})'
        return repr_str


def _crop_y_image(image, crop_ratio):
    image_height = image.shape[-2]
    magnitude = int(crop_ratio * image_height)
    if magnitude >= 0:
        image = image[:, magnitude:, :]
    else:
        image = image[:, :magnitude, :]
    return image

    
def _crop_y_target(target, crop_ratio, image_height):
    magnitude = int(crop_ratio * image_height)
    for key, value in target.items():
        if value.type not in ['masks', 'bboxes', 'points']:
            continue
        if value.type == 'masks':
            if value.islist:
                if magnitude >= 0:
                    crop = [v[:, magnitude:, :] for v in value.value]
                else:
                    crop = [v[:, :magnitude, :] for v in value.value]
            else:
                if magnitude >= 0:
                    crop = value.value[:, magnitude:, :]
                else:
                    crop = value.value[:, :magnitude, :]
        if value.type == 'bboxes':
            crop = value.value
            if magnitude >= 0:
                crop[:, [1, 3]] = torch.clamp(crop[:, [1, 3]] - magnitude, 0, image_height - 1)
            else:
                crop[:, [1, 3]] = torch.clamp(crop[:, [1, 3]], 0, image_height - 1)
            flag = (crop[:, 3] - crop[:, 1]) > 1
            crop = crop[flag, :]
        if value.type == 'points':
            crop = value.value
            if magnitude >= 0:
                crop[:, 1] = torch.clamp(crop[:, 1] - magnitude, 0, image_height - 1)
            else:
                crop[:, 1] = torch.clamp(crop[:, 1], 0, image_height - 1)
            flag = (crop[:, 1] > 0) & (crop[:, 1] < image_height - 1)
            crop = crop[flag, :]
        value.value = crop
        target[key] = value
    return target


@TRANSFORMS.register_module()
class CropY(SpatialBase):
    """Crop images along y-axis (height), images size = (h, w).

    Args:
        crop_ratio  (float  | tuple[float, float] |  list[float, float] ):
            float: Ratio of cropping in the range of [0, 1].
                    When crop_ratio is not None, it crops an image to (crop_ratio*h, w).
            tuple[float, float]: Range of the ratio of cropping. Default is None.
                    When crop_ratio=[min_ratio, max_ratio] is None, it crops
                    an image to (ratio*h, w), in which ratio is randomly selected from [min_ratio, max_ratio].
            list[float, ... , float] : list of the ratio of cropping to be randomly chosen.
        When crop_ratio is positve, the upper portion of the image is cropped out, and the lower portion is kept.
        When crop_ratio is negative, the upper portion of the image is kept, and the lower portion is cropped out.
    """

    def __init__(self, crop_ratio):
        super().__init__()
        _input_check_value_range_set(crop_ratio)
        self.crop_ratio = crop_ratio

    def calculate_image(self, image):
        self.image_height = image.shape[-2]
        self.crop_ratio_sel = _input_get_value_range_set(self.crop_ratio)
        return _crop_y_image(image, self.crop_ratio_sel)

    def calculate_target(self, target):
        return _crop_y_target(target, self.crop_ratio_sel, self.image_height)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(crop_ratio={self.crop_ratio})'
        return repr_str


@TRANSFORMS.register_module()
class RandomCropY(SpatialBase):
    """Crop images along y-axis (height), images size = (h, w).

    Args:
        crop_prob (float): Probability of the image being cropped.
        crop_ratio  (float  | tuple[float, float] |  list[float, float] ):
            float: Ratio of cropping in the range of [0, 1].
                    When crop_ratio is not None, it crops an image to (crop_ratio*h, w).
            tuple[float, float]: Range of the ratio of cropping. Default is None.
                    When crop_ratio=[min_ratio, max_ratio] is None, it crops
                    an image to (ratio*h, w), in which ratio is randomly selected from [min_ratio, max_ratio].
            list[float, ... , float] : list of the ratio of cropping to be randomly chosen.
        When crop_ratio is positve, the upper portion of the image is cropped out, and the lower portion is kept.
        When crop_ratio is negative, the upper portion of the image is kept, and the lower portion is cropped out.
    """

    def __init__(self, prob, crop_ratio=None):
        super().__init__()
        _input_get_value_range_set(crop_ratio)
        self.prob = prob
        self.crop_ratio = crop_ratio

    def calculate_image(self, image):
        self.randomness = False
        if random.random() < self.prob:
            self.randomness = True
            self.image_height = image.shape[-2]
            self.crop_ratio_sel = _input_get_value_range_set(self.crop_ratio)
            image = _crop_y_image(image, self.crop_ratio_sel)
        return image

    def calculate_target(self, target):
        if self.randomness:
            target = _crop_y_target(target, self.crop_ratio_sel, self.image_height)
        return target

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'crop_ratio={self.crop_ratio})'
        return repr_str


def _fliplr_image(image):
    image = torch.flip(image, dims=(2,))
    return image
    

def _fliplr_target(target, image_width):
    for key, value in target.items():
        if value.type not in ['masks', 'bboxes', 'points']:
            continue
        if value.type == 'masks':
            if value.islist:
                flip = [torch.flip(v, dims=(2,)) for v in value.value]
            else:
                flip = torch.flip(value.value, dims=(2,))
        if value.type == 'bboxes':
            flip = value.value
            flip[:, [0, 2]] = image_width - flip[:, [0, 2]] - 1
        if value.type == 'points':
            flip = value.value
            flip[:, 0] = image_width - flip[:, 0] - 1
        value.value = flip
        target[key] = value
    return target


@TRANSFORMS.register_module()
class Fliplr(SpatialBase):
    """Flip image left-right.
    """
    def __init__(self):
        super().__init__()

    def calculate_image(self, image):
        self.image_width = image.shape[-1]
        return _fliplr_image(image)

    def calculate_target(self, target):
        return _fliplr_target(target, self.image_width)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '()'
        return repr_str


@TRANSFORMS.register_module()
class RandomFliplr(SpatialBase):
    """Flip image left-right.

    Args:
        prob (float): Probability of flipping an image in the range of [0, 1].
                Default is None.
    """

    def __init__(self, prob: float = None):
        super().__init__()
        self.prob = prob

    def calculate_image(self, image):
        self.randomness = False
        if random.random() < self.prob:
            self.randomness = True
            self.image_width = image.shape[-1]
            image = _fliplr_image(image)
        return image

    def calculate_target(self, target):
        if self.randomness:
            target = _fliplr_target(target, self.image_width)
        return target

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob})'
        return repr_str


def _flipud_image(image):
    image = torch.flip(image, dims=(1,))
    return image


def _flipud_target(target, image_height):
    for key, value in target.items():
        if value.type not in ['masks', 'bboxes', 'points']:
            continue
        if value.type == 'masks':
            if value.islist:
                flip = [torch.flip(v, dims=(1,)) for v in value.value]
            else:
                flip = torch.flip(value.value, dims=(1,))
        if value.type == 'bboxes':
            flip = value.value
            flip[:, [1, 3]] = image_height - flip[:, [1, 3]] - 1
        if value.type == 'points':
            flip = value.value
            flip[:, 1] = image_height - flip[:, 1] - 1
        value.value = flip
        target[key] = value
    return target


@TRANSFORMS.register_module()
class Flipud(SpatialBase):
    """Flip image up-down.
    """

    def __init__(self):
        super().__init__()

    def calculate_image(self, image):
        self.image_height = image.shape[-2]
        return _flipud_image(image)
    
    def calculate_target(self, target):
        return _flipud_target(target, self.image_height)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '()'
        return repr_str


@TRANSFORMS.register_module()
class RandomFlipud(SpatialBase):
    """Flip image up-down.

    Args:
        prob (float): Probability of flipping an image in the range of [0, 1].
    """

    def __init__(self, prob: float = None):
        super().__init__()
        self.prob = prob

    def calculate_image(self, image):
        self.randomness = False
        if random.random() < self.prob:
            self.randomness = False
            image = _flipud_image(image)
        return image

    def calculate_target(self, target):
        if self.randomness:
            target = _flipud_target(target)
        return target

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob})'
        return repr_str


def _rotate_image(image, rotate_degree, fill):
    fill = _format_fill(fill, image)
    image = F.rotate(image, rotate_degree,  fill=fill)
    return image

    
def _rotate_target(target, rotate_degree, fill):
    for key, value in target.items():
        if value.type not in ['masks', 'bboxes', 'points']:
            continue
        if value.type == 'masks':
            if value.islist:
                rotate = [F.rotate(v, rotate_degree, interpolation=InterpolationMode.NEAREST, fill=0)
                          for v in value.value]
            else:
                rotate = F.rotate(value.value, rotate_degree, interpolation=InterpolationMode.NEAREST, fill=0)
        if value.type in ['bboxes', 'points']:
            raise ValueError(f"rotate does not support type={value.type}.")
        value.value = rotate
        target[key] = value
    return target


@TRANSFORMS.register_module()
class Rotate(SpatialBase):
    """Rotate image.

    Args:
        rotate_degree  (float  | tuple[float, float] |  list[float, float] ):
            float: Rotation angle value in degrees, counter-clockwise.
            tuple[float, float]: Range of rotation angle. Default is None.
            list[float, ... , float] : list of the ratio of rotation angle to be randomly chosen.
        fill (sequence or number, optional): Pixel fill value for the area outside the
                transformed image. If given a number, the value is used for all bands respectively.
    """
    def __init__(self, rotate_degree: float,
                 fill: Optional[List[float]] = None):
        super().__init__()
        _input_check_value_range_set(rotate_degree)
        self.rotate_degree = rotate_degree
        self.fill = fill

    def calculate_image(self, image):
        self.rotate_degree_sel = _input_get_value_range_set(self.rotate_degree)
        return _rotate_image(image, self.rotate_degree_sel, self.fill)

    def calculate_target(self, target):
        return _rotate_target(target, self.rotate_degree_sel, self.fill)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(rotate_degree={self.rotate_degree}, '
        repr_str += f'fill={self.fill})'
        return repr_str


@TRANSFORMS.register_module()
class RandomRotate(SpatialBase):
    """Rotate image.

    Args:
        rotate_degree  (float  | tuple[float, float] |  list[float, float] ):
            float: Rotation angle value in degrees, counter-clockwise.
            tuple[float, float]: Range of rotation angle. Default is None.
            list[float, ... , float] : list of the ratio of rotation angle to be randomly chosen.
        fill (sequence or number, optional): Pixel fill value for the area outside the
                transformed image. If given a number, the value is used for all bands respectively.
    """

    def __init__(self, prob: float, rotate_degree=None,
                 fill: Optional[List[float]] = None):
        super().__init__()
        _input_check_value_range_set(rotate_degree)
        self.prob = prob
        self.rotate_degree = rotate_degree
        self.fill = fill

    def calculate_image(self, image):
        self.randomness = False
        if random.random() < self.prob:
            self.randomness = True
            self.rotate_degree_sel = _input_get_value_range_set(self.rotate_degree)
            image = _rotate_image(image, self.rotate_degree_sel, self.fill)
        return image

    def calculate_target(self, target):
        if self.randomness:
            target = _rotate_target(target, self.rotate_degree_sel, self.fill)
        return target

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'rotate_degree={self.rotate_degree}, '
        repr_str += f'fill={self.fill})'
        return repr_str


@TRANSFORMS.register_module()
class Scale(SpatialBase):
    """Scale image according to scale_range.

    Args:
        scale_range (float | tuple[float, float]): expected image size scaling factor.
        scale_width (boolean): If scale_width=True, the width of image will be scaled to width * scale_width_factor
        scale_height (boolean): If scale_height=True, the height of image will be scaled to height * scale_height_factor
        scale_same (boolean): If scale_same=True, scaled factors will be generated by ``random.uniform`` according to
                                the scale_range separately. Otherwise, one factor will be generated and two factors
                                share the same value.
        interpolation (InterpolationMode): Desired interpolation enum defined by torchvision.transforms.InterpolationMode.
                                            Default is InterpolationMode.BILINEAR.
    """

    def __init__(self, scale_range=(0.8, 1.2),
                 scale_width=True, scale_height=False, scale_same=False,
                 interpolation: InterpolationMode = InterpolationMode.BILINEAR):
        super().__init__()
        _input_check_value_range_set(scale_range)
        self.scale_range = scale_range
        self.scale_width = scale_width
        self.scale_height = scale_height
        self.scale_same = scale_same
        self.interpolation = interpolation

    def calculate_image(self, image):
        scale1 = _input_get_value_range_set(self.scale_range)
        if self.scale_same:
            scale2 = scale1
        else:
            scale2 = _input_get_value_range_set(self.scale_range)
        width_factor = scale1 if self.scale_width else 1
        height_factor = scale2 if self.scale_height else 1
        self.factor = (height_factor, width_factor)
        h, w = image.shape[1], image.shape[2]
        image = F.resize(image, (int(h * height_factor), int(w * width_factor)), interpolation=self.interpolation)
        return image

    def calculate_target(self, target):
        height_factor, width_factor = self.factor
        for key, value in target.items():
            if value.type not in ['masks', 'bboxes', 'points']:
                continue
            value_scale = value.value.clone()
            if value.type == 'masks':
                if value.islist:
                    value_scale = [F.resize(v, (int(v.shape[1] * height_factor), int(v.shape[2] * width_factor)),
                                              interpolation=self.interpolation) for v in value_scale]
                else:
                    new_shape = (
                        int(value_scale.shape[1] * height_factor), int(value_scale.shape[2] * width_factor))
                    value_scale = F.resize(value_scale, new_shape, interpolation=self.interpolation)
            elif value.type == 'points':
                value_scale[:, 0] = value_scale[:, 0] * width_factor
                value_scale[:, 1] = value_scale[:, 1] * height_factor
            elif value.type == 'bboxes':
                value_scale[:, [0, 2]] = value_scale[:, [0, 2]] * width_factor
                value_scale[:, [1, 3]] = value_scale[:, [1, 3]] * height_factor
            value.value = value_scale
            target[key] = value
        return target

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(scale_range={self.scale_range}, '
        repr_str += f'(scale_width={self.scale_width}, '
        repr_str += f'(scale_height={self.scale_height}, '
        repr_str += f'(scale_same={self.scale_same})'
        repr_str += f'interpolation={self.interpolation})'
        return repr_str


@TRANSFORMS.register_module()
class RandomScale(SpatialBase):
    """Randomly scale image according to scale_range.

    Args:
        prob (float): Probability of applying this operation. Default is 0.5.
        scale_range (float | tuple[float, float]): expected image size scaling factor.
        scale_width (boolean): If scale_width=True, the width of image will be scaled to width * scale_width_factor
        scale_height (boolean): If scale_height=True, the height of image will be scaled to height * scale_height_factor
        scale_same (boolean): If scale_same=True, scaled factors will be generated by ``random.uniform`` according to
                                the scale_range separately. Otherwise, one factor will be generated and two factors
                                share the same value.
        interpolation (InterpolationMode): Desired interpolation enum defined by torchvision.transforms.InterpolationMode.
                                            Default is InterpolationMode.BILINEAR.
    """
    def __init__(self, prob=0.5, scale_range=(0.8, 1.2),
                 scale_width=True, scale_height=False, scale_same=False,
                 interpolation: InterpolationMode = InterpolationMode.BILINEAR):
        super().__init__()
        _input_check_value_range_set(scale_range)
        self.prob = prob
        self.scale_range = scale_range
        self.scale_width = scale_width
        self.scale_height = scale_height
        self.scale_same = scale_same
        self.interpolation = interpolation

    def calculate_image(self, image):
        self.randomness = False
        if random.random() < self.prob:
            self.randomness =  True
            scale1 = _input_get_value_range_set(self.scale_range)
            if self.scale_same:
                scale2 = scale1
            else:
                scale2 = _input_get_value_range_set(self.scale_range)
            width_factor = scale1 if self.scale_width else 1
            height_factor = scale2 if self.scale_height else 1
            self.factor = (height_factor, width_factor)
            h, w = image.shape[1], image.shape[2]
            image = F.resize(image, (int(h * height_factor), int(w * width_factor)),
                             interpolation=self.interpolation)
        return image

    def calculate_target(self, target):
        if self.randomness:
            height_factor, width_factor = self.factor
            for key, value in target.items():
                if value.type not in ['masks', 'bboxes', 'points']:
                    continue
                value_scale = value.value.clone()
                if value.type == 'masks':
                    if value.islist:
                        value_scale = [F.resize(v, (int(v.shape[1] * height_factor), int(v.shape[2] * width_factor)),
                                                  interpolation=self.interpolation) for v in value_scale]
                    else:
                        new_shape = (
                            int(value_scale.shape[1] * height_factor), int(value_scale.shape[2] * width_factor))
                        value_scale = F.resize(value_scale, new_shape, interpolation=self.interpolation)
                    if len(value_scale.shape) == 2:
                        value_scale = value_scale[..., None]
                elif value.type == 'points':
                    value_scale[:, 0] = value_scale[:, 0] * width_factor
                    value_scale[:, 1] = value_scale[:, 1] * height_factor
                elif value.type == 'bboxes':
                    value_scale[:, [0, 2]] = value_scale[:, [0, 2]] * width_factor
                    value_scale[:, [1, 3]] = value_scale[:, [1, 3]] * height_factor
                value.value = value_scale
                target[key] = value
        return target

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'(scale_range={self.scale_range}, '
        repr_str += f'(scale_width={self.scale_width}, '
        repr_str += f'(scale_height={self.scale_height}, '
        repr_str += f'(scale_same={self.scale_same})'
        repr_str += f'interpolation={self.interpolation})'
        return repr_str


def _resize_keypoints(keypoints, original_size, new_size):
    ratios = [
        torch.tensor(s, dtype=torch.float32, device=keypoints.device) /
        torch.tensor(s_orig, dtype=torch.float32, device=keypoints.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratio_h, ratio_w = ratios
    resized_data = keypoints.clone()
    resized_data[..., 0] *= ratio_w
    resized_data[..., 1] *= ratio_h
    return resized_data


def _resize_bboxes(bboxes, original_size, new_size):
    ratios = [
        torch.tensor(s, dtype=torch.float32, device=bboxes.device) /
        torch.tensor(s_orig, dtype=torch.float32, device=bboxes.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratio_height, ratio_width = ratios
    resized_bboxes = bboxes.clone()
    resized_bboxes[..., 0] *= ratio_width
    resized_bboxes[..., 1] *= ratio_height
    resized_bboxes[..., 2] *= ratio_width
    resized_bboxes[..., 3] *= ratio_height
    return resized_bboxes


@TRANSFORMS.register_module()
class Resize(SpatialBase):
    """Resize images.

    When `size` is int, it resizes an image to (size, size).
    When `size` is tuple/list and its second value is -1, it resizes an image
    such that the short edge of the resized image is equal to first value of `size`.
    Images scales for resizing (h, w).

    Args:
        size (int | tuple | list):
                When size is int, it resizes an image to (size, size).
                When size is tuple/list and the second value is -1, it resizes an image
                such that the short edge of the resized image is equal to first value of size.
                the short edge of an image is resized to its first value.
                For example, when size is 224, the image is resized to 224x224.
                When size is (224, -1), the short side is resized to 224 and the
                other side is computed based on the short side, maintaining the
                aspect ratio.
        interpolation (InterpolationMode): Desired interpolation enum defined by
                torchvision.transforms.InterpolationMode. Default is InterpolationMode.BILINEAR.
        max_size (int, optional): The maximum allowed for the longer edge of the resized image.
        fill (sequence or number, optional): Pixel fill value for the area outside the
                transformed image. If given a number, the value is used for all bands respectively.
    """

    def __init__(self, size: List[int], interpolation: InterpolationMode = InterpolationMode.BILINEAR,
                 max_size: Optional[int] = None, antialias: Optional[bool] = None):
        super().__init__()
        assert isinstance(size, int) or (isinstance(size, tuple) and len(size) == 2) or (
                isinstance(size, list) and len(size) == 2)
        self.resize_w_short_side = False
        if isinstance(size, int):
            assert size > 0
            size = (size, size)
        else:
            assert size[0] > 0 and (size[1] > 0 or size[1] == -1)
            if size[1] == -1:
                self.resize_w_short_side = True

        self.size = size
        self.interpolation = interpolation
        self.max_size = max_size
        self.antialias = antialias
        self.ignore_resize = False

    def calculate_image(self, image):        
        h, w = image.shape[-2:]
        self.image_size = (h, w)
        if self.resize_w_short_side:
            short_side = self.size[0]
            if (w <= h and w == short_side) or (h <= w and h == short_side):
                self.ignore_resize = True
            else:
                if w < h:
                    width = short_side
                    height = int(short_side * h / w)
                else:
                    height = short_side
                    width = int(short_side * w / h)
        else:
            height, width = self.size
        self.size_sel = (height, width)
        if not self.ignore_resize:
            image = F.resize(image, (height, width), self.interpolation, self.max_size, self.antialias)
        self.image_size_resize = image.shape[-2:]
        return image

    def calculate_target(self, target):
        if not self.ignore_resize:
            height, width = self.size_sel
            for key, value in target.items():
                if value.type not in ['masks', 'bboxes', 'points']:
                    continue
                if value.type == 'masks':
                    if value.islist:
                        value_resize = [F.resize(v, (height, width), interpolation=self.interpolation) for v in
                                        value.value]
                    else:
                        value_resize = F.resize(value.value, (height, width), interpolation=self.interpolation)
                elif value.type == 'points':
                    value_resize = _resize_keypoints(value.value, self.image_size, self.image_size_resize)
                elif value.type == 'bboxes':
                    value_resize = _resize_bboxes(value.value, self.image_size, self.image_size_resize)
                value.value = value_resize
                target[key] = value
        return target

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'interpolation={self.interpolation})'
        return repr_str


@TRANSFORMS.register_module()
class ResizeWidth(SpatialBase):
    """Resize images.

    When `width` is not None, it resizes the image to have width of `width`.
    When `width` is None, it resizes an image such that its shorter edge is `min_size_list[k]`,
    where `k` is randomly selected when the corresponding longer edge of the resized image is
    less than `max_size`; otherwise, it resizes an image such that the longer edge of the
    resized image is `max_size`.
    Images scales for resizing (h, w).

    Args:
        width (int): The width of the resized image.
                When width is not None, it resizes an image to (width*h/w, width).
                When width is None, it resizes an image according to min_size_list and max_size.
        min_size_list (list[int]): List of widths considered for resizing image. Default is None.
        max_size (int): The maximum allowed for the longer edge of the resized image.
                Default is None.
                When neither min_size_list and max_size are not None, it resizes an image such that its
                shorter edge is min_size_list[k] when the long edge is less than max_size, where k is
                picked randomly; otherwise its long edge is max_size.
        size (int | tuple | list):
                When size is int, the default behavior is to resize an image
                to (size, size). When size is tuple/list and the second value is -1,
                the short edge of an image is resized to its first value.
                For example, when size is 224, the image is resized to 224x224.
                When size is (224, -1), the short side is resized to 224 and the
                other side is computed based on the short side, maintaining the
                aspect ratio.
        interpolation (InterpolationMode): Desired interpolation enum defined by
                torchvision.transforms.InterpolationMode. Default is InterpolationMode.BILINEAR.
        fill (sequence or number, optional): Pixel fill value for the area outside the
                transformed image. If given a number, the value is used for all bands respectively.
    """

    def __init__(self, width: int = None, min_size_list: List[int] = None, max_size: int = None,
                 training: bool = False,
                 interpolation: InterpolationMode = InterpolationMode.BILINEAR,
                 antialias: Optional[bool] = None):
        super().__init__()
        self.width = width
        self.min_size_list = min_size_list
        self.max_size = max_size
        if (self.min_size_list is not None) & (self.max_size is not None):
            assert self.width is None, "width has to be None when min_size_list and max_size are given."
        else:
            assert self.width is not None, "width has to be given when min_size_list or max_size is None."
        self.interpolation = interpolation
        self.antialias = antialias
        self.training = training

    def calculate_image(self, image):
        h, w = image.shape[-2:]
        self.image_size_org = (h, w)
        if self.width is None:
            if self.training:
                self_min_size = float(self.torch_choice(self.min_size_list))
            else:
                self_min_size = float(self.min_size_list[-1])
            self_max_size = float(self.max_size)
            im_shape = torch.tensor(image.shape[-2:])
            min_size = float(torch.min(im_shape))
            max_size = float(torch.max(im_shape))
            scale_factor = self_min_size / min_size
            if max_size * scale_factor > self_max_size:
                scale_factor = self_max_size / max_size
        else:
            scale_factor = self.width / w
        self.size_sel = (int(scale_factor * h), int(scale_factor * w))
        image = F.resize(image, self.size_sel, self.interpolation, self.antialias)
        self.image_size_resize = image.shape[-2:]
        return image

    def calculate_target(self, target):
        for key, value in target.items():
            if value.type not in ['masks', 'bboxes', 'points']:
                continue
            if value.type == 'masks':
                if value.islist:
                    value_resize = [F.resize(v, self.size_sel, interpolation=self.interpolation, ) for v in value.value]
                else:
                    value_resize = F.resize(value.value, self.size_sel, interpolation=self.interpolation)
            elif value.type == 'points':
                value_resize = _resize_keypoints(value.value, self.image_size_org, self.image_size_resize)
            elif value.type == 'bboxes':
                value_resize = _resize_bboxes(value.value, self.image_size_org, self.image_size_resize)
            value.value = value_resize
            target[key] = value
        return target

    def torch_choice(self, k):
        index = int(torch.empty(1).uniform_(0., float(len(k))).item())
        return k[index]

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(width={self.width}, '
        repr_str += f'(min_size_list={self.min_size_list}, '
        repr_str += f'(max_size={self.max_size}, '
        repr_str += f'interpolation={self.interpolation})'
        return repr_str


@TRANSFORMS.register_module()
class RandomResizedCrop(SpatialBase):
    """Crop a random portion of image and resize it to a given size.

    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    A crop of the original image is made: the crop has a random area (H * W)
    and a random aspect ratio. This crop is finally resized to the given
    size. This is popularly used to train the Inception networks.

    Args:
        size (int | tule[int, int]): expected output size of the crop, for each edge. If size is an
            int instead of sequence like (h, w), a square output size ``(size, size)`` is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
        scale (tuple[float, float]): Specifies the lower and upper bounds for the random area of the crop,
            before resizing. The scale is defined with respect to the area of the original image.
        ratio (tuple[float, float]): lower and upper bounds for the random aspect ratio of the crop, before
            resizing.
        interpolation (InterpolationMode): interpolation (InterpolationMode): Desired interpolation enum defined by
                torchvision.transforms.InterpolationMode. Default is InterpolationMode.BILINEAR.
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=InterpolationMode.BILINEAR):
        super().__init__()
        self.size = setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")

        if not isinstance(scale, Sequence):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, Sequence):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(
            img: Tensor, scale: List[float], ratio: List[float]
    ) -> Tuple[int, int, int, int]:
        width, height = F.get_image_size(img)
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def calculate_image(self, image):
        self.loc = self.get_params(image, self.scale, self.ratio)
        i, j, h, w = self.loc
        image = F.resized_crop(image, i, j, h, w, self.size, self.interpolation)
        return image

    def calculate_target(self, target):
        i, j, h, w = self.loc
        for key, value in target.items():
            if value.type not in ['masks', 'bboxes', 'points']:
                continue
            if value.type == 'masks':
                if value.islist:
                    value_resize = [F.resized_crop(v, i, j, h, w, self.size, self.interpolation)
                                    for v in value.value]
                else:
                    value_resize = F.resized_crop(value.value, i, j, h, w, self.size, self.interpolation)
            if value.type in ['bboxes', 'points']:
                raise ValueError(f"RandomResizedCrop does not support type={value.type}.")
            value.value = value_resize
            target[key] = value
        return target

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(scale={self.scale}, '
        repr_str += f'ratio={self.ratio}, '
        repr_str += f'interpolation={self.interpolation})'
        return repr_str


@TRANSFORMS.register_module()
class RandomCrop(torch.nn.Module):
    """Randomly crop image.

    Args:
        crop_size (tule[int, int]): expected output size of the crop, for each edge. If size is an
            int instead of sequence like (h, w), a square output size ``(size, size)`` is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
        prob (float): The probability of cropping from the foreground.
        mask_type (str): The type of mask. It has to be in None, 'bboxes', or 'masks'. Default is None.
                When mask_type=None, it randomly crops from the image region.
                When mask_type='bboxes', it randomly crops from the foreground of the mask with probability=prob,
                and from the background of the mask with probability=1-prob. The mask is generated by bboxes of target.
                When mask_type='masks', it randomly crops from the foreground of the mask with probability=prob,
                and from the background of the mask with probability=1-prob. The mask is generated by masks of target.
        obj_labels (list): Foreground index in masks. It is only valid when mask_type='masks'.
                Default is None.
    """

    def __init__(self, crop_size: Tuple[int, int], prob: float,
                 mask_type: str = None, obj_labels: List = None):
        super().__init__()
        self.prob = prob  # prob to crop from object region
        self.crop_size = setup_size(crop_size, error_msg="Please provide only two dimensions (h, w) for size.")
        self.mask_type = mask_type
        self.obj_labels = obj_labels
        assert mask_type in [None, 'bboxes', 'masks']

    def forward(self, image, target):
        assert target is not None, "target can not be None, it has to include bboxes or masks for cropping based on RoI"
        if self.mask_type == 'bboxes':
            mask = image.new_full(image.shape[1:], 0, device=image.device)
            for key in target.keys():
                if target[key].type == 'bboxes':
                    box = target[key].value.int()
                    for k in range(box.shape[0]):
                        mask[box[k, 1]:box[k, 3] + 1, box[k, 0]:box[k, 2] + 1] = 1
                    break
        elif self.mask_type == 'masks':
            for key in target.keys():
                if target[key].type == 'masks':
                    mask = target[key].value
                    mask = torch.sum(mask[self.obj_labels], dim=0)
                    break
        else:
            mask = torch.ones(image.shape[-2:], device=image.device, dtype=image.dtype)
        crop_loc = self.crop_region(mask > 0.5)
        image = image[:, crop_loc[0]:crop_loc[1], crop_loc[2]:crop_loc[3]]

        for key, value in target.items():
            if value.type is None or value.type == 'labels':
                continue
            if value.type == 'masks':
                crop = value.value[:, crop_loc[0]:crop_loc[1], crop_loc[2]:crop_loc[3]]
            elif value.type == 'bboxes':
                h, w = image.shape[-2:]
                crop = value.value
                crop[:, 0] = torch.clamp(crop[:, 0] - crop_loc[2], 0, w - 1)
                crop[:, 1] = torch.clamp(crop[:, 1] - crop_loc[0], 0, h - 1)
                crop[:, 2] = torch.clamp(crop[:, 2] - crop_loc[2], 0, w - 1)
                crop[:, 3] = torch.clamp(crop[:, 3] - crop_loc[0], 0, h - 1)
                flag = (crop[:, 3] - crop[:, 1] > 1) & (crop[:, 2] - crop[:, 0] > 1)
                crop = crop[flag, :]
            elif value.type == 'points':
                h, w = image.shape[-2:]
                crop = value.value
                crop[:, 0] = torch.clamp(crop[:, 0] - crop_loc[2], 0, w - 1)
                crop[:, 1] = torch.clamp(crop[:, 1] - crop_loc[0], 0, h - 1)
                flag = (crop[:, 0] > 0) & (crop[:, 0] < w - 1) & (crop[:, 1] > 1) & (crop[:, 1] < h - 1)
                crop = crop[flag, :]
            value.value = crop
            target[key] = value
        return image, target

    def crop_region(self, mask):
        bw = mask
        if (self.mask_type is not None) & (random.random() >= self.prob):
            bw = mask == False  ## crop from background
        xl, yl = self.crop_size[0] // 2, self.crop_size[1] // 2
        xh, yh = self.crop_size[0] - xl, self.crop_size[1] - yl
        ind = torch.nonzero(bw, as_tuple=True)
        if len(ind[0]) == 0:
            x, y = bw.shape[0] // 2, bw.shape[1] // 2
        else:
            loc = torch.randint(high=len(ind[0]), size=(1,))
            x, y = ind[0][loc], ind[1][loc]
        xmin, ymin = x - xl, y - yl
        xmax, ymax = x + xh, y + yh
        if xmin < 0:
            xmin, xmax = 0, self.crop_size[0]
        if ymin < 0:
            ymin, ymax = 0, self.crop_size[1]
        if xmax >= bw.shape[0]:
            xmin, xmax = bw.shape[0] - self.crop_size[0], bw.shape[0]
        if ymax >= bw.shape[1]:
            ymin, ymax = bw.shape[1] - self.crop_size[1], bw.shape[1]

        return [xmin, xmax, ymin, ymax]

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(crop_size={self.crop_size}, '
        repr_str += f'(prob={self.prob}, '
        repr_str += f'(mask_type={self.mask_type}, '
        repr_str += f'(obj_labels={self.obj_labels})'
        return repr_str


@TRANSFORMS.register_module()
class ImagePadding(SpatialBase):
    """Pad image.

    It pads the shorter edge of an image to get a square padded image.
    """

    def __init__(self):
        super().__init__()

    def calculate_image(self, image):
        c, h, w = image.shape
        self.image_size = (h, w)
        if h > w:
            p = (h - w) // 2
            image_pad = torch.zeros((c, h, h), dtype=image.dtype, device=image.device)
            image_pad[:, :, p:p + w] = image
        elif h < w:
            p = (w - h) // 2
            image_pad = torch.zeros((c, w, w), dtype=image.dtype, device=image.device)
            image_pad[:, p:p + h, :] = image
        else:
            image_pad = image
        return image_pad

    def calculate_target(self, target):
        h, w = self.image_size
        if h > w:
            p = (h - w) // 2
            for key, value in target.items():
                if value.type not in ['masks', 'bboxes', 'points']:
                    continue
                if value.type == 'masks':
                    if value.islist:
                        pad = []
                        for v in value.value:
                            tmp = torch.zeros((v.shape[0], h, h), dtype=v.dtype, device=v.device)
                            tmp[:, :, p:p + w] = v
                        pad.append(tmp)
                    else:
                        pad = torch.zeros((value.value.shape[0], h, h), dtype=value.value.dtype,
                                          device=value.value.device)
                        pad[:, :, p:p + w] = value.value
                if value.type == 'bboxes':
                    pad = value.value
                    pad[:, [0, 2]] += p
                if value.type == 'points':
                    pad = value.value
                    pad[:, 0] += p
                value.value = pad
                target[key] = value
        elif h < w:
            p = (w - h) // 2
            for key, value in target.items():
                if value.type not in ['masks', 'bboxes', 'points']:
                    continue
                if value.type == 'masks':
                    if value.islist:
                        pad = []
                        for v in value.value:
                            tmp = torch.zeros((v.shape[0], w, w), dtype=v.dtype, device=v.device)
                            tmp[:, p:p + h, :] = v
                        pad.append(tmp)
                    else:
                        pad = torch.zeros((value.value.shape[0], w, w), dtype=value.value.dtype,
                                          device=value.value.device)
                        pad[:, p:p + h, :] = value.value
                if value.type == 'bboxes':
                    pad = value.value
                    pad[:, [1, 3]] += p
                if value.type == 'points':
                    pad = value.value
                    pad[:, 1] += p
                value.value = pad
                target[key] = value
        return target

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@TRANSFORMS.register_module()
class ImageHeightPaddingOrCrop(SpatialBase):
    """Pad or crop image along y-axis (height).

    It pads the y-axis of an image to get a square padded image, the x-axis is unchanged.
    """

    def __init__(self):
        super().__init__()

    def calculate_image(self, image):
        c, h, w = image.shape
        self.image_size = (h, w)
        if h > w:  # crop
            p = (h - w) // 2
            image_pad = image[:, p:p + w, :]
        elif h < w:  # padding
            p = (w - h) // 2
            image_pad = torch.zeros((c, w, w), dtype=image.dtype, device=image.device)
            image_pad[:, p:p + h, :] = image
        else:
            image_pad = image
        return image_pad

    def calculate_target(self, target):
        h, w = self.image_size
        if h > w:  # crop
            p = (h - w) // 2
            for key, value in target.items():
                if value.type not in ['masks', 'bboxes', 'points']:
                    continue
                if value.type == 'masks':
                    if value.islist:
                        pad = [v[:, p:p + w, :] for v in value.value]
                    else:
                        pad = value.value[:, p:p + w, :]
                if value.type == 'bboxes':
                    pad = value.value
                    pad[:, [1, 3]] -= p
                if value.type == 'points':
                    pad = value.value
                    pad[:, 1] -= p
                value.value = pad
                target[key] = value
        elif h < w:  # padding
            p = (w - h) // 2
            for key, value in target.items():
                if value.type not in ['masks', 'bboxes', 'points']:
                    continue
                if value.type == 'masks':
                    if value.islist:
                        pad = []
                        for v in value.value:
                            tmp = torch.zeros((v.shape[0], w, w), dtype=v.dtype, device=v.device)
                            tmp[:, p:p + h, :] = v
                        pad.append(v)
                    else:
                        pad = torch.zeros((value.value.shape[0], w, w), dtype=value.value.dtype,
                                          device=value.value.device)
                        pad[:, p:p + h, :] = value.value
                if value.type == 'bboxes':
                    pad = value.value
                    pad[:, [1, 3]] += p
                if value.type == 'points':
                    pad = value.value
                    pad[:, 1] += p
                value.value = pad
                target[key] = value      
        return target

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str