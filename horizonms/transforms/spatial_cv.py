import random
from turtle import width
import cv2
import math
import warnings
import numbers
import numpy as np
from collections.abc import Sequence
from .utils import cv_image_shift, _input_check_value_range_set, _input_get_value_range_set
from typing import List, Optional, Tuple, Union
from abc import ABC, abstractmethod
from ..builder import TRANSFORMS
from .utils import setup_size


__all__ = ["CVSpatialBase", "CVShearX", "CVShearY", "CVTranslateX", "CVTranslateY",
           "CVCropX", "CVCropY", "CVFliplr", "CVFlipud", "CVRotate", "CVScale",
           "CVResize", "CVResizeWidth", "CVRandomResizedCrop", "CVRandomCrop",
           "CVImagePadding",  "CVRandomShift", "CVRandomShearX", "CVRandomShearY",
           "CVRandomTranslateX", "CVRandomTranslateY", "CVRandomCropX",
           "CVRandomCropY", "CVRandomFliplr", "CVRandomFlipud", "CVRandomRotate",
           "CVRandomScale", "CVRandomMaskCrop"
]


class CVSpatialBase(ABC):
    """Base for spatial operators implemented by OpenCV.
    """
    @abstractmethod
    def calculate_image(self, image):
        """conduct transformation for image.

        Args:
            image (np.array): image data with dimension HxWxC.
        """
        pass

    @abstractmethod
    def calculate_target(self, target):
        """conduct transformation for target.

        Args:
            target (Dict): target data in dictionary format.
        """
        pass

    def __call__(self, image, target=None):
        """implement transformation for image and/or target.

        Args:
            image (np.array): image data with dimension HxWxC.
            target (Dict): target data in dictionary format. Default: `None`.
        """
        image = self.calculate_image(image)
        if target is None:
            return image
        else:
            target = self.calculate_target(target)
            return image, target


IMG_ROW_AXIS = 0
IMG_COL_AXIS = 1
IMG_CHANNEL_AXIS = 2


def _get_dimensions_cv(img) -> List[int]:
    channels = 1 if len(img.shape) == 2 else img.shape[IMG_CHANNEL_AXIS]
    height, width = img.shape[:2]
    return [height, width, channels]


def _get_inverse_affine_matrix_cv(
        center: List[float], angle: float, translate: List[float], scale: float, shear: List[float],
        inverted: bool = True
) -> List[float]:
    # Helper method to compute inverse matrix for affine transformation

    # Pillow requires inverse affine transformation matrix:
    # Affine matrix is : M = T * C * RotateScaleShear * C^-1
    #
    # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
    #       C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
    #       RotateScaleShear is rotation with scale and shear matrix
    #
    #       RotateScaleShear(a, s, (sx, sy)) =
    #       = R(a) * S(s) * SHy(sy) * SHx(sx)
    #       = [ s*cos(a - sy)/cos(sy), s*(-cos(a - sy)*tan(sx)/cos(sy) - sin(a)), 0 ]
    #         [ s*sin(a + sy)/cos(sy), s*(-sin(a - sy)*tan(sx)/cos(sy) + cos(a)), 0 ]
    #         [ 0                    , 0                                      , 1 ]
    # where R is a rotation matrix, S is a scaling matrix, and SHx and SHy are the shears:
    # SHx(s) = [1, -tan(s)] and SHy(s) = [1      , 0]
    #          [0, 1      ]              [-tan(s), 1]
    #
    # Thus, the inverse is M^-1 = C * RotateScaleShear^-1 * C^-1 * T^-1

    rot = math.radians(angle)
    sx = math.radians(shear[0])
    sy = math.radians(shear[1])

    cx, cy = center
    tx, ty = translate

    # RSS without scaling
    a = math.cos(rot - sy) / math.cos(sy)
    b = -math.cos(rot - sy) * math.tan(sx) / math.cos(sy) - math.sin(rot)
    c = math.sin(rot - sy) / math.cos(sy)
    d = -math.sin(rot - sy) * math.tan(sx) / math.cos(sy) + math.cos(rot)

    if inverted:
        # Inverted rotation matrix with scale and shear
        # det([[a, b], [c, d]]) == 1, since det(rotation) = 1 and det(shear) = 1
        matrix = [d, -b, 0.0, -c, a, 0.0]
        matrix = [x / scale for x in matrix]
        # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
        matrix[2] += matrix[0] * (-cx - tx) + matrix[1] * (-cy - ty)
        matrix[5] += matrix[3] * (-cx - tx) + matrix[4] * (-cy - ty)
        # Apply center translation: C * RSS^-1 * C^-1 * T^-1
        matrix[2] += cx
        matrix[5] += cy
    else:
        matrix = [a, b, 0.0, c, d, 0.0]
        matrix = [x * scale for x in matrix]
        # Apply inverse of center translation: RSS * C^-1
        matrix[2] += matrix[0] * (-cx) + matrix[1] * (-cy)
        matrix[5] += matrix[3] * (-cx) + matrix[4] * (-cy)
        # Apply translation and center : T * C * RSS * C^-1
        matrix[2] += cx + tx
        matrix[5] += cy + ty

    return matrix


def _format_fill_cv(fill, image):
    if isinstance(fill, (int, float)):
        fill = [float(fill)] * _get_dimensions_cv(image)[IMG_CHANNEL_AXIS]
    elif fill is not None:
        fill = [float(f) for f in fill]
    return fill


def _affine_cv(image, angle=0.0, translate=[0, 0], scale=1.0,
               shear=[0.0, 0.0], fill=(0.0, 0.0, 0.0)):
    rows = image.shape[0]
    cols = image.shape[1]

    img_center = (cols / 2, rows / 2)

    matrix = _get_inverse_affine_matrix_cv(img_center, angle, translate, scale, shear)
    matrix = np.reshape(matrix, (2, 3))
    image = cv2.warpAffine(image, matrix, (cols, rows), borderValue=fill)
    if len(image.shape) == 2:
        image = image[..., None]
    return image


def _shear_x_image(image, shear_degree, fill):
    fill = _format_fill_cv(fill, image)
    image = _affine_cv(image, angle=0.0,
                       translate=[0, 0],
                       scale=1.0,
                       shear=[shear_degree, 0.0],
                       fill=fill)
    return image


def _shear_x_target(target, shear_degree, fill):
    for key, value in target.items():
        if value.type not in ['masks', 'bboxes', 'points']:
            continue
        if value.type == 'masks':
            if value.islist:
                shear = [_affine_cv(v, angle=0.0, translate=[0, 0], scale=1.0,
                                    shear=[shear_degree, 0.0],
                                    fill=fill)
                         for v in value.value]
            else:
                shear = _affine_cv(value.value, angle=0.0, translate=[0, 0], scale=1.0,
                                   shear=[shear_degree, 0.0],
                                   fill=fill)
        if value.type in ['bboxes', 'points']:
            raise ValueError(f"shear does not support type={value.type}.")
        value.value = shear
        target[key] = value
    return target


@TRANSFORMS.register_module()
class CVShearX(CVSpatialBase):
    """Shear image along x-axis (width).

    Args:
        shear_degree (float | tuple[float, float] | list[float]): shear angle in degree between -180 and 180, clockwise direction.
            There are three ways for shear angle as follows:
                - If `shear_degree` is `float`, then share angle is the value.
                - If `shear_degree` is `tuple[float, float]` (i.e. an angle range), then shear angle is randomly selected from the range.
                - If `shear_degree` is `list[float, ... , float]` (i.e. list of angles), then shear angle is randomly selected from the list.
        fill (sequence or number, optional): pixel fill value for the area outside the
            transformed image. If given a number, the value is used for all bands respectively.
            Default: `None`.
    """
    def __init__(self, shear_degree: Union[float, Tuple[float], List[float]],
                 fill: Optional[List[float]] = None):
        super().__init__()
        _input_check_value_range_set(shear_degree)
        self.shear_degree = shear_degree
        self.fill = fill

    def calculate_image(self, image):
        self.shear_degree_sel = 180 - _input_get_value_range_set(self.shear_degree)
        return _shear_x_image(image, self.shear_degree_sel, self.fill)

    def calculate_target(self, target):
        return _shear_x_target(target, self.shear_degree_sel, self.fill)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(shear_degree={self.shear_degree}, '
        repr_str += f'fill={self.fill})'
        return repr_str


@TRANSFORMS.register_module()
class CVRandomShearX(CVSpatialBase):
    """Randomly shear image along x-axis (width) with a predefined probability.

    Args:
        prob (float): probability of the image being sheared.
        shear_degree (float | tuple[float, float] | list[float]): shear angle in degree between -180 and 180, clockwise direction.
            There are three ways for shear angle as follows:
                - If `shear_degree` is `float`, then share angle is the value.
                - If `shear_degree` is `tuple[float, float]` (i.e. an angle range), then shear angle is randomly selected from the range.
                - If `shear_degree` is `list[float, ... , float]` (i.e. list of angles), then shear angle is randomly selected from the list.
        fill (sequence or number, optional): pixel fill value for the area outside the
            transformed image. If given a number, the value is used for all bands respectively.
            Default: `None`.
    """
    def __init__(self, prob: float, shear_degree: Union[float, Tuple[float], List[float]],
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
            self.shear_degree_sel = 180 - _input_get_value_range_set(self.shear_degree)
            return _shear_x_image(image, self.shear_degree_sel, self.fill)
        return image
    
    def calculate_target(self, target):
        if self.randomness:
            target = _shear_x_target(target, self.shear_degree_sel, self.fill)
        return target

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'shear_degree={self.shear_degree}, '
        repr_str += f'fill={self.fill})'
        return repr_str


def _shear_y_image(image, shear_degree, fill):
    fill = _format_fill_cv(fill, image)
    image = _affine_cv(image, angle=0.0,
                       translate=[0, 0],
                       scale=1.0,
                       shear=[0.0, shear_degree],
                       fill=fill)
    return image


def _shear_y_target(target, shear_degree, fill):
    for key, value in target.items():
        if value.type not in ['masks', 'bboxes', 'points']:
            continue
        if value.type == 'masks':
            if value.islist:
                shear = [_affine_cv(v, angle=0.0, translate=[0, 0], scale=1.0,
                                    shear=[0.0, shear_degree],
                                    fill=fill)
                         for v in value.value]
            else:
                shear = _affine_cv(value.value, angle=0.0, translate=[0, 0], scale=1.0,
                                   shear=[0.0, shear_degree],
                                   fill=fill)
        if value.type in ['bboxes', 'points']:
            raise ValueError(f"shear does not support type={value.type}.")
        value.value = shear
        target[key] = value
    return target


@TRANSFORMS.register_module()
class CVShearY(CVSpatialBase):
    """Shear image along y-axis (height).

    Args:
        shear_degree (float | tuple[float, float] | list[float]): shear angle in degree between -180 and 180, clockwise direction.
            There are three ways for shear angle as follows:
                - If `shear_degree` is `float`, then share angle is the value.
                - If `shear_degree` is `tuple[float, float]` (i.e. an angle range), then shear angle is randomly selected from the range.
                - If `shear_degree` is `list[float, ... , float]` (i.e. list of angles), then shear angle is randomly selected from the list.
        fill (sequence or number, optional): pixel fill value for the area outside the
            transformed image. If given a number, the value is used for all bands respectively.
            Default: `None`.
    """
    def __init__(self, shear_degree: Union[float, Tuple[float], List[float]],
                 fill: Optional[List[float]] = None):
        super().__init__()
        _input_check_value_range_set(shear_degree)
        self.shear_degree = shear_degree
        self.fill = fill

    def calculate_image(self, image):
        self.shear_degree_sel = 180 - _input_get_value_range_set(self.shear_degree)
        return _shear_y_image(image, self.shear_degree_sel, self.fill)

    def calculate_target(self, target):
        return _shear_y_target(target, self.shear_degree_sel, self.fill)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(shear_degree={self.shear_degree}, '
        repr_str += f'fill={self.fill})'
        return repr_str


@TRANSFORMS.register_module()
class CVRandomShearY(CVSpatialBase):
    """Randomly shear image along y-axis (height) with a predefined probability.

    Args:
        prob (float): probability of the image being sheared.
        shear_degree (float | tuple[float, float] | list[float]): shear angle in degree between -180 and 180, clockwise direction.
            There are three ways for shear angle as follows:
                - If `shear_degree` is `float`, then share angle is the value.
                - If `shear_degree` is `tuple[float, float]` (i.e. an angle range), then shear angle is randomly selected from the range.
                - If `shear_degree` is `list[float, ... , float]` (i.e. list of angles), then shear angle is randomly selected from the list.
        fill (sequence or number, optional): pixel fill value for the area outside the
            transformed image. If given a number, the value is used for all bands respectively.
            Default: `None`.
    """
    def __init__(self, prob: float, shear_degree: Union[float, Tuple[float], List[float]],
                 fill: Optional[List[float]] = None):
        super().__init__()
        self.prob = prob
        self.shear_degree = shear_degree
        self.fill = fill

    def calculate_image(self, image):
        self.randomness = False
        if random.random() < self.prob:
            self.randomness = True
            self.shear_degree_sel = 180 - _input_get_value_range_set(self.shear_degree)
            image = _shear_y_image(image, self.shear_degree_sel, self.fill)
        return image

    def calculate_target(self, target):
        if self.randomness:
            target = _shear_y_target(target, self.shear_degree_sel, self.fill)
        return target

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'shear_degree={self.shear_degree}, '
        repr_str += f'fill={self.fill})'
        return repr_str


def _translate_x_image(image, translate_ratio, fill):
    fill = _format_fill_cv(fill, image)
    magnitude = int(translate_ratio * image.shape[IMG_COL_AXIS])
    image = _affine_cv(
        image,
        angle=0.0,
        translate=[magnitude, 0],
        scale=1.0,
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
                translate = [_affine_cv(v, angle=0.0, translate=[magnitude, 0],
                                        scale=1.0, shear=[0.0, 0.0]
                                        )
                             for v in value.value]
            else:
                translate = _affine_cv(value.value, angle=0.0, translate=[magnitude, 0],
                                       scale=1.0, shear=[0.0, 0.0]
                                       )
        if value.type == 'bboxes':
            translate = value.value

            translate[:, [0, 2]] = np.clip(translate[:, [0, 2]] - magnitude, 0, image_width - 1)
            flag = (translate[:, 2] - translate[:, 0]) > 1
            translate = translate[flag, :]
        if value.type == 'points':
            translate = value.value
            translate[:, 0] = np.clip(translate[:, 0] - magnitude, 0, image_width - 1)
            flag = (translate[:, 0] > 0) & (translate[:, 0] < image_width - 1)
            translate = translate[flag, :]
        value.value = translate
        target[key] = value
    return target


@TRANSFORMS.register_module()
class CVTranslateX(CVSpatialBase):
    """Translate image along x-axis (width).

    Args:
        translate_ratio (float | tuple[float, float] | list[float]): ratio of translation in range between 0 and 1.
            There are three ways for translation ratio as follows:
                - If `translate_ratio` is `float`, then ratio of translation is the value.
                - If `translate_ratio` is `tuple[float, float]` (i.e. a ratio range), then ratio of translation is randomly selected from the range.
                - If `translate_ratio` is `list[float, ... , float]` (i.e. list of angles), then ratio of translation is randomly selected from the list.
        fill (sequence or number, optional): pixel fill value for the area outside the
            transformed image. If given a number, the value is used for all bands respectively.
            Default: `None`.
    
    When `translate_ratio` is positve, the image moves to the right.    
    When `translate_ratio` is negative, the image moves to the left.
    """
    def __init__(self, translate_ratio: Union[float, Tuple[float], List[float]],
                 fill: Optional[List[float]] = None):
        super().__init__()
        _input_check_value_range_set(translate_ratio)
        self.translate_ratio = translate_ratio
        self.fill = fill

    def calculate_image(self, image):
        self.image_width = image.shape[IMG_COL_AXIS]
        self.translate_ratio_sel = -1 * _input_get_value_range_set(self.translate_ratio)
        return _translate_x_image(image, self.translate_ratio_sel, self.fill)

    def calculate_target(self, target):
        return _translate_x_target(target, self.translate_ratio_sel, self.image_width)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(translate_ratio={self.translate_ratio}, '
        repr_str += f'fill={self.fill})'
        return repr_str


@TRANSFORMS.register_module()
class CVRandomTranslateX(CVSpatialBase):
    """Randomly translate image along x-axis (width) with a predefined probability.

    Args:
        prob (float): probability of the image being translated.
        translate_ratio (float | tuple[float, float] | list[float]): ratio of translation in range between 0 and 1.
            There are three ways for translation ratio as follows:
                - If `translate_ratio` is `float`, then ratio of translation is the value.
                - If `translate_ratio` is `tuple[float, float]` (i.e. a ratio range), then ratio of translation is randomly selected from the range.
                - If `translate_ratio` is `list[float, ... , float]` (i.e. list of angles), then ratio of translation is randomly selected from the list.
        fill (sequence or number, optional): pixel fill value for the area outside the
            transformed image. If given a number, the value is used for all bands respectively.
            Default: `None`.

    When `translate_ratio` is positve, the image moves to the right.
    When `translate_ratio` is negative, the image moves to the left.
    """
    def __init__(self, prob: float, translate_ratio: Union[float, Tuple[float], List[float]],
                 interpolation=None,
                 fill: Optional[List[float]] = None):
        super().__init__()
        self.prob = prob
        self.translate_ratio = translate_ratio
        self.fill = fill

    def calculate_image(self, image):
        self.randomness = False
        if random.random() < self.prob:
            self.randomness = True
            self.image_width = image.shape[IMG_COL_AXIS]
            self.translate_ratio_sel = -1 * _input_get_value_range_set(self.translate_ratio)
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
    fill = _format_fill_cv(fill, image)
    magnitude = int(translate_ratio * image.shape[IMG_ROW_AXIS])
    image = _affine_cv(
        image,
        angle=0.0,
        translate=[0, magnitude],
        scale=1.0,
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
                translate = [_affine_cv(v, angle=0.0, translate=[0, magnitude],
                                        scale=1.0, shear=[0.0, 0.0])
                             for v in value.value]
            else:
                translate = _affine_cv(value.value, angle=0.0, translate=[0, magnitude],
                                       scale=1.0, shear=[0.0, 0.0])
        if value.type == 'bboxes':
            translate = value.value
            translate[:, [1, 3]] = np.clip(translate[:, [1, 3]] - magnitude, 0, image_height - 1)
            flag = (translate[:, 3] - translate[:, 1]) > 1
            translate = translate[flag, :]
        if value.type == 'points':
            translate = value.value
            translate[:, 1] = np.clip(translate[:, 1] - magnitude, 0, image_height - 1)
            flag = (translate[:, 1] > 0) & (translate[:, 1] < image_height - 1)
            translate = translate[flag, :]
        value.value = translate
        target[key] = value
    return target


@TRANSFORMS.register_module()
class CVTranslateY(CVSpatialBase):
    """Translate image along y-axis (height).

    Args:
        translate_ratio (float | tuple[float, float] | list[float]): ratio of translation in range between 0 and 1.
            There are three ways for translation ratio as follows:
                - If `translate_ratio` is `float`, then ratio of translation is the value.
                - If `translate_ratio` is `tuple[float, float]` (i.e. a ratio range), then ratio of translation is randomly selected from the range.
                - If `translate_ratio` is `list[float, ... , float]` (i.e. list of angles), then ratio of translation is randomly selected from the list.
        fill (sequence or number, optional): pixel fill value for the area outside the
            transformed image. If given a number, the value is used for all bands respectively.
            Default: `None`.

    When `translate_ratio` is positve, the image moves to the right.
    When `translate_ratio` is negative, the image moves to the left.
    """
    def __init__(self, translate_ratio: Union[float, Tuple[float], List[float]],
                 fill: Optional[List[float]] = None):
        super().__init__()
        _input_check_value_range_set(translate_ratio)
        self.translate_ratio = translate_ratio
        self.fill = fill

    def calculate_image(self, image):
        self.image_height = image.shape[IMG_ROW_AXIS]
        self.translate_ratio_sel = -1 * _input_get_value_range_set(self.translate_ratio)
        return _translate_y_image(image, self.translate_ratio_sel, self.fill)

    def calculate_target(self, target):
        return _translate_y_target(target, self.translate_ratio_sel, self.image_height)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(translate_ratio={self.translate_ratio}, '
        repr_str += f'fill={self.fill})'
        return repr_str


@TRANSFORMS.register_module()
class CVRandomTranslateY(CVSpatialBase):
    """Randomly translate image along y-axis (height) with a predefined probability. 

    Args:
        prob (float): probability of the image being translated.
        translate_ratio (float | tuple[float, float] | list[float]): ratio of translation in range between 0 and 1.
            There are three ways for translation ratio as follows:
                - If `translate_ratio` is `float`, then ratio of translation is the value. 
                - If `translate_ratio` is `tuple[float, float]` (i.e. a ratio range), then ratio of translation is randomly selected from the range.
                - If `translate_ratio` is `list[float, ... , float]` (i.e. list of angles), then ratio of translation is randomly selected from the list.
        fill (sequence or number, optional): pixel fill value for the area outside the
            transformed image. If given a number, the value is used for all bands respectively.
            Default: `None`.

    When `translate_ratio` is positve, the image moves to the right.
    When `translate_ratio` is negative, the image moves to the left.
    """
    def __init__(self, prob: float, translate_ratio: Union[float, Tuple[float], List[float]],

                 fill: Optional[List[float]] = None):
        super().__init__()
        self.prob = prob
        self.translate_ratio = translate_ratio
        self.fill = fill

    def calculate_image(self, image):
        self.randomness = False
        if random.random() < self.prob:
            self.randomness =  True
            self.image_height = image.shape[IMG_ROW_AXIS]
            self.translate_ratio_sel = -1 * _input_get_value_range_set(self.translate_ratio)
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
    magnitude = int(crop_ratio * image.shape[IMG_COL_AXIS])
    if magnitude >= 0:
        image = image[:, magnitude:, :]
    else:
        image = image[:, :magnitude, :]
    return image


def _crop_x_target(target, crop_ratio, image_width):
    magnitude = int(crop_ratio * image_width)
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
                crop[:, [0, 2]] = np.clip(crop[:, [0, 2]] - magnitude, 0, image_width - 1)
            else:
                crop[:, [0, 2]] = np.clip(crop[:, [0, 2]], 0, image_width - 1)
            flag = (crop[:, 2] - crop[:, 0]) > 1
            crop = crop[flag, :]
        if value.type == 'points':
            crop = value.value
            if magnitude >= 0:
                crop[:, 0] = np.clip(crop[:, 0] - magnitude, 0, image_width - 1)
            else:
                crop[:, 0] = np.clip(crop[:, 0], 0, image_width - 1)
            flag = (crop[:, 0] > 0) & (crop[:, 0] < image_width - 1)
            crop = crop[flag, :]
        value.value = crop
        target[key] = value
    return target


@TRANSFORMS.register_module()
class CVCropX(CVSpatialBase):
    """Crop image along x-axis (width).

    Args:
        crop_ratio (float | tuple[float, float] | list[float]): ratio of cropping in range between 0 and 1.
            There are three ways for ratio of cropping as follows:
                - If `crop_ratio` is `float`, then ratio of cropping is the value.
                - If `crop_ratio` is `tuple[float, float]` (i.e. a ratio range), then ratio of cropping is randomly selected from the range.
                - If `crop_ratio` is `list[float, ... , float]` (i.e. list of angles), then ratio of cropping is randomly selected from the list.
        
    When `crop_ratio` is positve, the left portion of the image is cropped out, and the right portion is kept.
    When `crop_ratio` is negative, the left portion of the image is kept, and the right portion is cropped out.
    """
    def __init__(self, crop_ratio: Union[float, Tuple[float], List[float]]):
        super().__init__()
        _input_check_value_range_set(crop_ratio)
        self.crop_ratio = crop_ratio

    def calculate_image(self, image):
        self.image_width = image.shape[IMG_COL_AXIS]
        self.crop_ratio_sel = _input_get_value_range_set(self.crop_ratio)
        return _crop_x_image(image, self.crop_ratio_sel)

    def calculate_target(self, target):
        return _crop_x_target(target, self.crop_ratio_sel, self.image_width)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(crop_ratio={self.crop_ratio})'
        return repr_str


@TRANSFORMS.register_module()
class CVRandomCropX(CVSpatialBase):
    """Randomly crop image along x-axis (width). with a predefined probability.

    Args:
        prob (float): probability of the image being cropped.
        crop_ratio (float | tuple[float, float] | list[float]): ratio of cropping in range between 0 and 1.
            There are three ways for ratio of cropping as follows:
                - If `crop_ratio` is `float`, then ratio of cropping is the value.
                - If `crop_ratio` is `tuple[float, float]` (i.e. a ratio range), then ratio of cropping is randomly selected from the range.
                - If `crop_ratio` is `list[float, ... , float]` (i.e. list of angles), then ratio of cropping is randomly selected from the list.
        
    When `crop_ratio` is positve, the left portion of the image is cropped out, and the right portion is kept.
    When `crop_ratio` is negative, the left portion of the image is kept, and the right portion is cropped out.
    """
    def __init__(self, prob: float, crop_ratio: Union[float, Tuple[float], List[float]]):
        super().__init__()
        _input_check_value_range_set(crop_ratio)
        self.prob = prob
        self.crop_ratio = crop_ratio

    def calculate_image(self, image):
        self.randomness = False
        if random.random() < self.prob:
            self.randomness = True
            self.image_width = image.shape[IMG_COL_AXIS]
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
    magnitude = int(crop_ratio * image.shape[IMG_ROW_AXIS])
    if magnitude >= 0:
        image = image[magnitude:, :, :]
    else:
        image = image[:magnitude, :, :]
    return image

def _crop_y_target(target, crop_ratio, image_height):
    magnitude = int(crop_ratio * image_height)
    for key, value in target.items():
        if value.type not in ['masks', 'bboxes', 'points']:
            continue
        if value.type == 'masks':
            if value.islist:
                if magnitude >= 0:
                    crop = [v[magnitude:, :, :] for v in value.value]
                else:
                    crop = [v[:magnitude, :, :] for v in value.value]
            else:
                if magnitude >= 0:
                    crop = value.value[magnitude:, :, :]
                else:
                    crop = value.value[:magnitude, :, :]
        if value.type == 'bboxes':
            crop = value.value
            if magnitude >= 0:
                crop[:, [1, 3]] = np.clip(crop[:, [1, 3]] - magnitude, 0, image_height - 1)
            else:
                crop[:, [1, 3]] = np.clip(crop[:, [1, 3]], 0, image_height - 1)
            flag = (crop[:, 3] - crop[:, 1]) > 1
            crop = crop[flag, :]
        if value.type == 'points':
            crop = value.value
            if magnitude >= 0:
                crop[:, 1] = np.clip(crop[:, 1] - magnitude, 0, image_height - 1)
            else:
                crop[:, 1] = np.clip(crop[:, 1], 0, image_height - 1)
            flag = (crop[:, 1] > 0) & (crop[:, 1] < image_height - 1)
            crop = crop[flag, :]
        value.value = crop
        target[key] = value
    return target


@TRANSFORMS.register_module()
class CVCropY(CVSpatialBase):
    """Crop image along y-axis (height).

    Args:
        crop_ratio (float | tuple[float, float] | list[float]): ratio of cropping in range between 0 and 1.
            There are three ways for ratio of cropping as follows:
                - If `crop_ratio` is `float`, then ratio of cropping is the value.
                - If `crop_ratio` is `tuple[float, float]` (i.e. a ratio range), then ratio of cropping is randomly selected from the range.
                - If `crop_ratio` is `list[float, ... , float]` (i.e. list of angles), then ratio of cropping is randomly selected from the list.
        
    When `crop_ratio` is positve, the upper portion of the image is cropped out, and the lower portion is kept.
    When `crop_ratio` is negative, the upper portion of the image is kept, and the lower portion is cropped out.
    """
    def __init__(self, crop_ratio: Union[float, Tuple[float], List[float]]):
        super().__init__()
        _input_check_value_range_set(crop_ratio)
        self.crop_ratio = crop_ratio

    def calculate_image(self, image):
        self.image_height = image.shape[IMG_ROW_AXIS]
        self.crop_ratio_sel = _input_get_value_range_set(self.crop_ratio)
        return _crop_y_image(image, self.crop_ratio_sel)

    def calculate_target(self, target):
        return _crop_y_target(target, self.crop_ratio_sel, self.image_height)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(crop_ratio={self.crop_ratio})'
        return repr_str


@TRANSFORMS.register_module()
class CVRandomCropY(CVSpatialBase):
    """Randomly crop image along y-axis (height). with a predefined probability.

    Args:
        crop_prob (float): probability of the image being cropped.
        crop_ratio (float | tuple[float, float] | list[float]): ratio of cropping in range between 0 and 1.
            There are three ways for ratio of cropping as follows:
                - If `crop_ratio` is `float`, then ratio of cropping is the value.
                - If `crop_ratio` is `tuple[float, float]` (i.e. a ratio range), then ratio of cropping is randomly selected from the range.
                - If `crop_ratio` is `list[float, ... , float]` (i.e. list of angles), then ratio of cropping is randomly selected from the list.
        
    When `crop_ratio` is positve, the upper portion of the image is cropped out, and the lower portion is kept.
    When `crop_ratio` is negative, the upper portion of the image is kept, and the lower portion is cropped out.
    """
    def __init__(self, prob: float, crop_ratio: Union[float, Tuple[float], List[float]]):
        super().__init__()
        self.prob = prob
        self.crop_ratio = crop_ratio

    def calculate_image(self, image):
        self.randomness = False
        if random.random() < self.prob:
            self.randomness = True
            self.image_height = image.shape[IMG_ROW_AXIS]
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
    image = np.fliplr(image).copy()
    return image


def _fliplr_target(target, image_width):
    for key, value in target.items():
        if value.type not in ['masks', 'bboxes', 'points']:
            continue
        value_flip = value.value.copy()
        if value.type == 'masks':
            if value.islist:
                value_flip = [np.fliplr(v) for v in value_flip]
            else:
                value_flip = np.fliplr(value_flip)
        elif value.type == 'points':
            value_flip[:, 0] = image_width - value_flip[:, 0]
        elif value.type == 'bboxes':
            xmin = image_width - value_flip[:, 2]
            xmax = image_width - value_flip[:, 0]
            value_flip[:, 0] = xmin
            value_flip[:, 2] = xmax
        value.value = value_flip
        target[key] = value
    return target


@TRANSFORMS.register_module()
class CVFliplr(CVSpatialBase):
    """Flip image left-right.
    """
    def __init__(self):
        super().__init__()

    def calculate_image(self, image):
        self.image_width = image.shape[IMG_COL_AXIS]
        return _fliplr_image(image)
    
    def calculate_target(self, target):
        return _fliplr_target(target, self.image_width)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '()'
        return repr_str


@TRANSFORMS.register_module()
class CVRandomFliplr(CVSpatialBase):
    """Randomly flip image left-right with a predefined probability.

    Args:
        prob (float): probability of flipping an image in the range of [0, 1].
    """
    def __init__(self, prob: float):
        self.prob = prob

    def calculate_image(self, image):
        self.randomness = False
        if random.random() < self.prob:
            self.randomness = True
            self.image_width = image.shape[IMG_COL_AXIS]
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
    image = np.flipud(image)
    return image

def _flipud_target(target, image_height):
    for key, value in target.items():
        if value.type not in ['masks', 'bboxes', 'points']:
            continue
        if value.type == 'masks':
            if value.islist:
                flip = [np.flipud(v) for v in value.value]
            else:
                flip = np.flipud(value.value)
        if value.type == 'bboxes':
            flip = value.value
            x_min, x_max = flip[:, [1]], flip[:, [3]]
            flip[:, [1]] = image_height - x_max - 1
            flip[:, [3]] = image_height - x_min - 1
        if value.type == 'points':
            flip = value.value
            flip[:, 1] = image_height - flip[:, 1] - 1
        value.value = flip
        target[key] = value
    return target


@TRANSFORMS.register_module()
class CVFlipud(CVSpatialBase):
    """Flip image up-down.
    """
    def __init__(self):
        super().__init__()

    def calculate_image(self, image):
        self.image_height = image.shape[IMG_ROW_AXIS]
        return _flipud_image(image)

    def calculate_target(self, target):
        return _flipud_target(target, self.image_height)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '()'
        return repr_str


@TRANSFORMS.register_module()
class CVRandomFlipud(CVSpatialBase):
    """Randomly flip image up-down with a predefined probability.

    Args:
        prob (float): probability of flipping an image in the range of [0, 1].
    """
    def __init__(self, prob: float):
        super().__init__()
        self.prob = prob

    def calculate_image(self, image):
        self.randomness = False
        if random.random() < self.prob:
            self.randomness = True
            self.image_height = image.shape[IMG_ROW_AXIS]
            image = _flipud_image(image)
        return image

    def calculate_target(self, target):
        if self.randomness:
            target = _flipud_target(target, self.image_height)
        return target

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob})'
        return repr_str


def _rotate_cv(image, angle=0.0, fill=(0.0, 0.0, 0.0)):
    rows = image.shape[0]
    cols = image.shape[1]

    matrix = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), angle, 1)
    image = cv2.warpAffine(image, matrix, (cols, rows), borderValue=fill)
    if len(image.shape) == 2:
        image = image[..., None]
    return image


def _rotate_image(image, rotate_degree, fill):
    fill = _format_fill_cv(fill, image)
    image = _rotate_cv(image, rotate_degree, fill=fill)
    return image


def _rotate_target(target, rotate_degree):
    for key, value in target.items():
        if value.type not in ['masks', 'bboxes', 'points']:
            continue
        if value.type == 'masks':
            if value.islist:
                rotate = [_rotate_cv(v, rotate_degree, fill=0)
                          for v in value.value]
            else:
                rotate = _rotate_cv(value.value, rotate_degree, fill=0)
        if value.type in ['bboxes', 'points']:
            raise ValueError(f"Rotate does not support type={value.type}.")
        value.value = rotate
        target[key] = value
    return target


@TRANSFORMS.register_module()
class CVRotate(CVSpatialBase):
    """Rotate image.

    Args:
        rotate_degree (float | tuple[float, float] | list[float]): angle of rotation in degree, counter-clockwise.
            There are three ways for angle of rotation as follows:
                - If `rotate_degree` is `float`, then angle of rotation is the value.
                - If `rotate_degree` is `tuple[float, float]` (i.e. a ratio range), then angle of rotation is randomly selected from the range.
                - If `rotate_degree` is `list[float, ... , float]` (i.e. list of angles), then angle of rotation is randomly selected from the list.
        fill (sequence or number, optional): Pixel fill value for the area outside the
            transformed image. If given a number, the value is used for all bands respectively.
            Default: `None`.
    """
    def __init__(self, rotate_degree: Union[float, Tuple[float], List[float]],
                 fill: Optional[List[float]] = None):
        super().__init__()
        _input_check_value_range_set(rotate_degree)
        self.rotate_degree = rotate_degree
        self.fill = fill

    def calculate_image(self, image):
        self.rotate_degree_sel = _input_get_value_range_set(self.rotate_degree)
        return _rotate_image(image, self.rotate_degree_sel, self.fill)

    def calculate_target(self, target):
        return _rotate_target(target, self.rotate_degree_sel)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(rotate_degree={self.rotate_degree}, '
        repr_str += f'fill={self.fill})'
        return repr_str


@TRANSFORMS.register_module()
class CVRandomRotate(CVSpatialBase):
    """Randomly rotate image with a predefined probability.

    Args:
        prob (float): probability of rotating an image in the range of [0, 1].
        rotate_degree (float | tuple[float, float] | list[float]): angle of rotation in degree, counter-clockwise.
            There are three ways for angle of rotation as follows:
                - If `rotate_degree` is `float`, then angle of rotation is the value.
                - If `rotate_degree` is `tuple[float, float]` (i.e. a ratio range), then angle of rotation is randomly selected from the range.
                - If `rotate_degree` is `list[float, ... , float]` (i.e. list of angles), then angle of rotation is randomly selected from the list.
        fill (sequence or number, optional): Pixel fill value for the area outside the
            transformed image. If given a number, the value is used for all bands respectively.
            Default: `None`.
    """
    def __init__(self, prob: float, rotate_degree: Union[float, Tuple[float], List[float]],
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
            target = _rotate_target(target, self.rotate_degree_sel)
        return target

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'rotate_degree={self.rotate_degree}, '
        repr_str += f'fill={self.fill})'
        return repr_str


def _resize_keypoints_cv(keypoints, original_size, new_size):
    ratios = [s / s_orig for s, s_orig in zip(new_size, original_size)]
    ratio_h, ratio_w = ratios
    resized_data = keypoints
    resized_data[..., 0] = ratio_w * resized_data[..., 0]
    resized_data[..., 1] = ratio_h * resized_data[..., 1]
    return resized_data


def _resize_bboxes_cv(bboxes, original_size, new_size):
    ratios = [s / s_orig for s, s_orig in zip(new_size, original_size)]
    ratio_height, ratio_width = ratios
    resized_bboxes = bboxes
    resized_bboxes[..., 0] = ratio_width * resized_bboxes[..., 0]
    resized_bboxes[..., 1] = ratio_height * resized_bboxes[..., 1]
    resized_bboxes[..., 2] = ratio_width * resized_bboxes[..., 2]
    resized_bboxes[..., 3] = ratio_height * resized_bboxes[..., 3]
    return resized_bboxes


@TRANSFORMS.register_module()
class CVScale(CVSpatialBase):
    """Scale image.

    Args:
        scale_factor (float | tuple[float, float] | list[float]): image scaling factor.
            There are three ways for scaling factor as follows:
                - If `scale_factor` is `float`, then scaling factor is the value.
                - If `scale_factor` is `tuple[float, float]` (i.e. a ratio range), then scaling factor is randomly selected from the range.
                - If `scale_factor` is `list[float, ... , float]` (i.e. list of angles), then scaling factor is randomly selected from the list.
        flag_scale_width (bool): if True, the width of image is scaled by scaling ratio; otherwise, no scaling is done for image width.
            Default: `True`.
        flag_scale_height (bool): if True, the height of image is scaled by scaling ratio; otherwise, no scaling is done for image height.
            Default: `False`.
        flag_scale_same (bool): if True, scaling factors for width and height are same; otherwise, they are different.
            Default: `False`.
        interpolation (InterpolationMode): interpolation enum defined by torchvision.transforms.InterpolationMode.
            Default: `InterpolationMode.BILINEAR`.
    """
    def __init__(self, scale_factor: Union[float, Tuple[float], List[float]],
                 flag_scale_width: bool = True, flag_scale_height: bool = False, 
                 flag_scale_same: bool = False,
                 interpolation='cv2.INTER_LINEAR'):
        _input_check_value_range_set(scale_factor)
        self.scale_factor = scale_factor
        self.flag_scale_width = flag_scale_width
        self.flag_scale_height = flag_scale_height
        self.flag_scale_same = flag_scale_same
        self.interpolation = eval(interpolation)

    def calculate_image(self, image):
        scale1 = _input_get_value_range_set(self.scale_factor)
        if self.flag_scale_same:
            scale2 = scale1
        else:
            scale2 = _input_get_value_range_set(self.scale_factor)
        width_factor = scale1 if self.flag_scale_width else 1
        height_factor = scale2 if self.flag_scale_height else 1
        self.factor = (height_factor, width_factor)
        h, w = image.shape[:2]
        image = cv2.resize(image, (int(w * width_factor), int(h * height_factor)), interpolation=self.interpolation)
        if len(image.shape) == 2:
            image = image[..., None]
        return image

    def calculate_target(self, target):
        height_factor, width_factor = self.factor
        for key, value in target.items():
            if value.type not in ['masks', 'bboxes', 'points']:
                continue
            value_scale = value.value.copy()
            if value.type == 'masks':
                if value.islist:
                    value_scale = [cv2.resize(v, (int(v.shape[1] * width_factor), int(v.shape[0] * height_factor)),
                                              interpolation=self.interpolation) for v in value_scale]
                    if len(value_scale[0].shape) == 2:
                        value_scale = [v[..., None] for v in value_scale]
                else:
                    new_shape = (
                        int(value_scale.shape[1] * width_factor), int(value_scale.shape[0] * height_factor))
                    value_scale = cv2.resize(value_scale, new_shape, interpolation=self.interpolation)
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
        repr_str += f'(scale_factor={self.scale_factor}, '
        repr_str += f'(flag_scale_width={self.flag_scale_width}, '
        repr_str += f'(flag_scale_height={self.flag_scale_height}, '
        repr_str += f'(flag_scale_same={self.flag_scale_same})'
        repr_str += f'interpolation={self.interpolation})'
        return repr_str


@TRANSFORMS.register_module()
class CVRandomScale(CVSpatialBase):
    """Randomly scale image with a predefined probability.

    Args:
        prob (float): probability of an image being scaled. 
        scale_factor (float | tuple[float, float] | list[float]): image scaling factor.
            There are three ways for scaling factor as follows:
                - If `scale_factor` is `float`, then scaling factor is the value.
                - If `scale_factor` is `tuple[float, float]` (i.e. a ratio range), then scaling factor is randomly selected from the range.
                - If `scale_factor` is `list[float, ... , float]` (i.e. list of angles), then scaling factor is randomly selected from the list.
        flag_scale_width (bool): if True, the width of image is scaled by scaling ratio; otherwise, no scaling is done for image width.
            Default: `True`.
        flag_scale_height (bool): if True, the height of image is scaled by scaling ratio; otherwise, no scaling is done for image height.
            Default: `False`.
        flag_scale_same (bool): if True, scaling factors for width and height are same; otherwise, they are different.
            Default: `False`.
        interpolation (InterpolationMode): Desired interpolation enum defined by torchvision.transforms.InterpolationMode.
            Default: `InterpolationMode.BILINEAR`.
    """
    def __init__(self, prob: float, scale_factor: Union[float, Tuple[float], List[float]],
                 flag_scale_width: bool = True, flag_scale_height: bool = False, 
                 flag_scale_same: bool = False,
                 interpolation='cv2.INTER_LINEAR'):
        _input_check_value_range_set(scale_factor)
        self.prob = prob
        self.scale_factor = scale_factor
        self.flag_scale_width = flag_scale_width
        self.flag_scale_height = flag_scale_height
        self.flag_scale_same = flag_scale_same
        self.interpolation = eval(interpolation)

    def calculate_image(self, image):
        self.randomness = False
        if random.random() < self.prob:
            self.randomness =  True
            scale1 = _input_get_value_range_set(self.scale_factor)
            if self.flag_scale_same:
                scale2 = scale1
            else:
                scale2 = _input_get_value_range_set(self.scale_factor)
            width_factor = scale1 if self.flag_scale_width else 1
            height_factor = scale2 if self.flag_scale_height else 1
            self.factor = (height_factor, width_factor)
            h, w = image.shape[:2]
            image = cv2.resize(image, (int(w * width_factor), int(h * height_factor)),
                               interpolation=self.interpolation)
            if len(image.shape) == 2:
                image = image[..., None]
        return image

    def calculate_target(self, target):
        if self.randomness:
            height_factor, width_factor = self.factor
            for key, value in target.items():
                if value.type not in ['masks', 'bboxes', 'points']:
                    continue
                value_scale = value.value.copy()
                if value.type == 'masks':
                    if value.islist:
                        value_scale = [cv2.resize(v, (int(v.shape[1] * width_factor), int(v.shape[0] * height_factor)),
                                                  interpolation=self.interpolation) for v in value_scale]
                        if len(value_scale[0].shape) == 2:
                            value_scale = [v[..., None] for v in value_scale]
                    else:
                        new_shape = (
                            int(value_scale.shape[1] * width_factor), int(value_scale.shape[0] * height_factor))
                        value_scale = cv2.resize(value_scale, new_shape, interpolation=self.interpolation)
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
        repr_str += f'(scale_factor={self.scale_factor}, '
        repr_str += f'(flag_scale_width={self.flag_scale_width}, '
        repr_str += f'(flag_scale_height={self.flag_scale_height}, '
        repr_str += f'(flag_scale_same={self.flag_scale_same})'
        repr_str += f'interpolation={self.interpolation})'
        return repr_str


@TRANSFORMS.register_module()
class CVResize(CVSpatialBase):
    def __init__(self, size, interpolation='cv2.INTER_LINEAR'):
        self.image_size = size
        self.interpolation = eval(interpolation)

    def calculate_image(self, image):
        height, width = image.shape[:2]
        self.image_size_org = (height, width)
        image = cv2.resize(image, (self.image_size[1],self.image_size[0]), interpolation=self.interpolation)
        if len(image.shape) == 2:
            image = image[..., None]
        return image

    def calculate_target(self, target):
        height, width = self.image_size_org
        ratio_height = self.image_size[0] / height
        ratio_width = self.image_size[1] / width
        for key, value in target.items():
            if value.type not in ['masks', 'bboxes', 'points']:
                continue
            value_resize = value.value.copy()
            if value.type == 'masks':
                if value.islist:
                    value_resize = [cv2.resize(v, (self.image_size[1],self.image_size[0]),
                                               interpolation=self.interpolation) for v in value_resize
                                    if (v.shape[0] == height) & (v.shape[1] == width)]
                    if len(value_resize[0].shape) == 2:
                        value_resize = [v[..., None] for v in value_resize]
                else:
                    value_resize = cv2.resize(value_resize, (self.image_size[1],self.image_size[0]),
                                              interpolation=self.interpolation)
                    if len(value_resize.shape) == 2:
                        value_resize = value_resize[..., None]

            elif value.type == 'points':
                value_resize[:, 0] = value_resize[:, 0] * ratio_width
                value_resize[:, 1] = value_resize[:, 1] * ratio_height
            elif value.type == 'bboxes':
                value_resize[:, [0, 2]] = value_resize[:, [0, 2]] * ratio_width
                value_resize[:, [1, 3]] = value_resize[:, [1, 3]] * ratio_height

            value.value = value_resize
            target[key] = value
        return target

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'interpolation={self.interpolation})'
        return repr_str


@TRANSFORMS.register_module()
class CVResizeWidth(CVSpatialBase):
    """Resize image such that its width is a target value.

    When `width` is not None, it resizes the image to have width of `width`.
    When `width` is None, it resizes an image such that its shorter edge is `min_size_list[k]`,
    where `k` is randomly selected when the corresponding longer edge of the resized image is
    less than `max_size`; otherwise, it resizes an image such that the longer edge of the
    resized image is `max_size`.

    Args:
        width (int): width of the resized image.
            When width is not None, it resizes an image to (width*h/w, width).
            When width is None, it resizes an image according to min_size_list and max_size.
        min_size_list (list[int]): list of widths considered for resizing image. Default: `None`.
        max_size (int):  maximum allowed for the longer edge of the resized image.
            Default: `None`.
            When neither min_size_list and max_size are not None, it resizes an image such that its
            shorter edge is min_size_list[k] when the long edge is less than max_size, where k is
            picked randomly; otherwise its long edge is max_size.
        training (bool): if True, width is randomly selected from `min_size_list`; otherwise, width is set as `min_size_list[-1]`.
            Default: `None`.
        interpolation (InterpolationMode): interpolation enum defined by
            torchvision.transforms.InterpolationMode. Default: `InterpolationMode.BILINEAR`.
    """
    def __init__(self, width: int = None, min_size_list: List[int] = None, max_size: int = None,
                 training: bool = False, interpolation: str = 'cv2.INTER_LINEAR'):
        super().__init__()
        self.width = width
        self.min_size_list = min_size_list
        self.max_size = max_size
        if (self.min_size_list is not None) & (self.max_size is not None):
            assert self.width is None, "width has to be None when min_size_list and max_size are given."
        else:
            assert self.width is not None, "width has to be given when min_size_list or max_size is None."
        self.interpolation = eval(interpolation)
        self.training = training

    def calculate_image(self, image):
        h, w = image.shape[IMG_ROW_AXIS], image.shape[IMG_COL_AXIS]
        self.image_size_org = (h, w)
        if self.width is None:

            self_min_size = float(self.min_size_list[-1])
            self_max_size = float(self.max_size)
            im_shape = image.shape[:2]
            min_size = float(np.min(im_shape))
            max_size = float(np.max(im_shape))
            scale_factor = self_min_size / min_size
            if max_size * scale_factor > self_max_size:
                scale_factor = self_max_size / max_size
        else:
            scale_factor = self.width / w
        self.size = (int(scale_factor * w), int(scale_factor * h))
        
        image = cv2.resize(image, self.size, interpolation=self.interpolation)
        if len(image.shape) == 2:
            image = image[..., None]
        self.image_size_resize = image.shape[:2]
        return image

    def calculate_target(self, target):
        for key, value in target.items():
            if value.type not in ['masks', 'bboxes', 'points']:
                continue
            if value.type == 'masks':
                if value.islist:
                    value_resize = [cv2.resize(v, self.size, interpolation=self.interpolation, ) for v in value.value]
                    if len(value_resize[0].shape) == 2:
                        value_resize = [v[..., None] for v in value_resize]

                else:
                    value_resize = cv2.resize(value.value, self.size, interpolation=self.interpolation)
                    if len(value_resize.shape) == 2:
                            value_resize = value_resize[..., None]
            elif value.type == 'points':
                value_resize = _resize_keypoints_cv(value.value, self.image_size_org, self.image_size_resize)
            elif value.type == 'bboxes':
                value_resize = _resize_bboxes_cv(value.value, self.image_size_org, self.image_size_resize)

            value.value = value_resize
            target[key] = value
        return target

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(width={self.width}, '
        repr_str += f'(min_size_list={self.min_size_list}, '
        repr_str += f'(max_size={self.max_size}, '
        repr_str += f'interpolation={self.interpolation})'
        return repr_str


def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size


def _resized_crop_cv(img, top, left, height, width, size, interpolation):
    img = img[top:top + height, left:left + width]
    img = cv2.resize(img, size, interpolation=interpolation)
    if len(img.shape) == 2:
        img = img[..., None]
    return img


@TRANSFORMS.register_module()
class CVRandomResizedCrop(CVSpatialBase):
    """Crop a random portion of image and resize it to a given size.

    A crop of the original image is made: the crop has a random area (H * W)
    and a random aspect ratio. This crop is finally resized to the given
    size. This is popularly used to train the Inception networks.

    Args:
        size (int | tule[int, int]): expected output size of the crop, for each edge. If size is an
            int instead of sequence like (h, w), a square output size ``(size, size)`` is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
        scale (tuple[float, float]): lower and upper bounds for the random area of the crop,
            before resizing. The scale is defined with respect to the area of the original image.
            Default: `(0.08, 1.0)`.
        ratio (tuple[float, float]): lower and upper bounds for the random aspect ratio of the crop, before
            resizing. Default: `(3. / 4., 4. / 3.)`.
        interpolation (str): Default: 'cv2.INTER_LINEAR'.
    """
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation='cv2.INTER_LINEAR'):
        super().__init__()
        self.size = _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")

        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")

        self.interpolation = eval(interpolation)
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(
            img, scale: List[float], ratio: List[float]
    ) -> Tuple[int, int, int, int]:
        height, width = img.shape[IMG_ROW_AXIS], img.shape[IMG_COL_AXIS]
        area = height * width

        log_ratio = np.log(ratio)
        for _ in range(10):
            target_area = area * np.random.uniform(scale[0], scale[1])
            aspect_ratio = np.exp(
                np.random.uniform(log_ratio[0], log_ratio[1])
            )

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = np.random.randint(0, height - h + 1)
                j = np.random.randint(0, width - w + 1)
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
        image = _resized_crop_cv(image, i, j, h, w, self.size, self.interpolation)
        return image

    def calculate_target(self, target):
        i, j, h, w = self.loc
        for key, value in target.items():
            if value.type not in ['masks', 'bboxes', 'points']:
                continue
            if value.type == 'masks':
                if value.islist:
                    value_resize = [_resized_crop_cv(v, i, j, h, w, self.size, self.interpolation)
                                    for v in value.value]
                else:
                    value_resize = _resized_crop_cv(value.value, i, j, h, w, self.size, self.interpolation)
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
class CVRandomCrop(CVSpatialBase):
    def __init__(self, prob: float = 0.5, crop_ratio: float = 0.6):
        self.prob = prob
        self.crop_ratio = crop_ratio

    def calculate_image(self, image):
        self.randomness = False
        if random.random() < self.prob:
            self.randomness = True
            height, width = image.shape[:2]
            self.image_size = (height, width)
            h = random.uniform(self.crop_ratio * height, height)
            w = random.uniform(self.crop_ratio * width, width)
            x = random.uniform(0, width - w)
            y = random.uniform(0, height - h)
            x, y, h, w = int(x), int(y), int(h), int(w)
            self.crop_sel = (x, y, h, w)
            image = image[y:y + h, x:x + w, :]
        return image

    def calculate_target(self, target):
        if self.randomness:
            x, y, h, w = self.crop_sel
            height, width = self.image_size
            for key, value in target.items():
                if value.type not in ['masks', 'bboxes', 'points']:
                    continue
                value_crop = value.value.copy()
                if value.type == 'masks':
                    if value.islist:
                        value_crop = [v[y:y + h, x:x + w, :] for v in value_crop
                                      if (v.shape[0] == height) & (v.shape[1] == width)]
                    else:
                        value_crop = value_crop[y:y + h, x:x + w, :]
                elif value.type == 'points':
                    value_crop[:, 0] = value_crop[:, 0] - x
                    value_crop[:, 1] = value_crop[:, 1] - y
                    flag1 = (value_crop[:, 0] >= 0) & (value_crop[:, 0] <= w - 1)
                    flag2 = (value_crop[:, 1] >= 0) & (value_crop[:, 1] <= h - 1)
                    value_crop = value_crop[flag1 & flag2, :]
                elif value.type == 'bboxes':
                    value_crop[:, [0, 2]] = value_crop[:, [0, 2]] - x
                    value_crop[:, [1, 3]] = value_crop[:, [1, 3]] - y
                    value_crop[:, [0, 2]] = value_crop[:, [0, 2]].clip(0, w)
                    value_crop[:, [1, 3]] = value_crop[:, [1, 3]].clip(0, h)
                    flag = (value_crop[:, 2] - value_crop[:, 0] > 1) & (value_crop[:, 3] - value_crop[:, 1] > 1)
                    value_crop = value_crop[flag, :]
                value.value = value_crop
                target[key] = value
        return target

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'crop_ratio={self.crop_ratio})'
        return repr_str


@TRANSFORMS.register_module()
class CVImagePadding(CVSpatialBase):
    """Pad the shorter edge of an image such that width and height are equal.
    """
    def __init__(self):
        super().__init__()

    def calculate_image(self, image):
        self.image_size = image.shape[:2]
        maxdim = np.max(self.image_size)
        image_pad = np.zeros((maxdim, maxdim, image.shape[-1]))
        image_pad[
        int((maxdim - self.image_size[0]) / 2):int((maxdim - self.image_size[0]) / 2) + image.shape[
            IMG_ROW_AXIS],
        int((maxdim - self.image_size[1]) / 2):int((maxdim - self.image_size[1]) / 2) + image.shape[
            IMG_COL_AXIS]] = image

        return image_pad
        
    def calculate_target(self, target):
        maxdim = np.max(self.image_size)
        h, w = self.image_size
        for key, value in target.items():
            if value.type not in ['masks', 'bboxes', 'points']:
                continue
            if value.type == 'masks':
                if value.islist:
                    pad = []
                    for v in value.value:
                        mask_pad = np.zeros((maxdim, maxdim, v.shape[IMG_CHANNEL_AXIS]))
                        mask_pad[
                        int((maxdim - h) / 2):int((maxdim - h) / 2) + v.shape[IMG_ROW_AXIS],
                        int((maxdim - w) / 2):int((maxdim - w) / 2) + v.shape[IMG_COL_AXIS]] = v
                    pad.append(mask_pad)
                else:
                    pad = np.zeros((maxdim, maxdim, value.value.shape[IMG_CHANNEL_AXIS]))
                    pad[int((maxdim - h) / 2):int((maxdim - h) / 2) + value.value.shape[IMG_ROW_AXIS],
                    int((maxdim - w) / 2):int((maxdim - w) / 2) + value.value.shape[IMG_COL_AXIS]] = value.value
            if value.type == 'bboxes':
                pad = value.value
                if h > w:
                    p = (h - w) // 2
                    pad[:, [0, 2]] += p
                else:
                    p = (w - h) // 2
                    pad[:, [1, 3]] += p
            if value.type == 'points':
                pad = value.value
                if h > w:
                    p = (h - w) // 2
                    pad[:, 0] += p
                else:
                    p = (w - h) // 2
                    pad[:, 1] += p
            value.value = pad
            target[key] = value
        return target

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '()'
        return repr_str


@TRANSFORMS.register_module()
class CVRandomShift(CVSpatialBase):
    """Randomly shift image along x-axis and y-axis with a predefined probability.

    Args:
        prob (float): probability of image shifting.
        shift_limit (float): maximum shift ratio along x-axis and y-axis.
            Default: `0.2`.
    """
    def __init__(self, prob: float, shift_limit: float = 0.2):
        self.prob = prob
        self.shift_limit = shift_limit

    def calculate_image(self, image):
        self.randomness = False
        if random.random() < self.prob:
            self.randomness = True
            image = image.astype(np.float32)
            height, width = image.shape[:2]
            self.image_size = (height, width)
            shift_x = random.uniform(-width * self.shift_limit, width * self.shift_limit)
            shift_y = random.uniform(-height * self.shift_limit, height * self.shift_limit)
            self.shift_sel = (shift_y, shift_x)
            image = cv_image_shift(image, (shift_x, shift_y))
        return image

    def calculate_target(self, target):
        if self.randomness:
            shift_y, shift_x = self.shift_sel
            height, width = self.image_size
            for key, value in target.items():
                if value.type not in ['masks', 'bboxes', 'points']:
                    continue
                value_shift = value.value.copy()
                if value.type == 'masks':
                    if value.islist:
                        value_shift = [cv_image_shift(v, (shift_x, shift_y)) for v in value_shift
                                       if (v.shape[0] == height) & (v.shape[1] == width)]
                    else:
                        value_shift = cv_image_shift(value_shift, (shift_x, shift_y))
                elif value.type == 'points':
                    value_shift[:, 0] = value_shift[:, 0] - shift_x
                    value_shift[:, 1] = value_shift[:, 1] - shift_y
                    flag1 = (value_shift[:, 0] >= 0) & (value_shift[:, 0] <= width - 1)
                    flag2 = (value_shift[:, 1] >= 0) & (value_shift[:, 1] <= height - 1)
                    value_shift = value_shift[flag1 & flag2, :]
                elif value.type == 'bboxes':
                    value_shift[:, [0, 2]] = value_shift[:, [0, 2]] + shift_x
                    value_shift[:, [1, 3]] = value_shift[:, [1, 3]] + shift_y
                    value_shift[:, [0, 2]] = value_shift[:, [0, 2]].clip(0, width)
                    value_shift[:, [1, 3]] = value_shift[:, [1, 3]].clip(0, height)
                    flag = (value_shift[:, 2] - value_shift[:, 0] > 1) & (value_shift[:, 3] - value_shift[:, 1] > 1)
                    value_shift = value_shift[flag, :]
                value.value = value_shift
                target[key] = value
        return target

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'(shift_limit={self.shift_limit})'
        return repr_str


@TRANSFORMS.register_module()
class CVRandomMaskCrop:
    """Randomly crop image.

    Args:
        crop_size (tule[int, int]): expected output size of the crop, for each edge. If size is an
            int instead of sequence like (h, w), a square output size ``(size, size)`` is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
        prob (float): probability of cropping from the foreground.
        mask_type (str): type of mask. It has to be in None, 'bboxes', or 'masks'. Default is None.
            When `mask_type=None`, it randomly crops from the image region.
            When `mask_type='bboxes'`, it randomly crops from the foreground of the mask with probability=prob,
            and from the background of the mask with probability=1-prob. The mask is generated by bboxes of target.
            When `mask_type='masks'`, it randomly crops from the foreground of the mask with probability=prob,
            and from the background of the mask with probability=1-prob. The mask is generated by masks of target.
            Default: `None`.
        obj_labels (list): foreground index in masks. It is only valid when mask_type='masks'.
                Default: `None`.
    """
    def __init__(self, crop_size: Tuple[int, int], prob: float,
                 mask_type: str = None, obj_labels: List = None):
        super().__init__()
        self.prob = prob  # prob to crop from object region
        self.crop_size = setup_size(crop_size, error_msg="Please provide only two dimensions (h, w) for size.")
        self.mask_type = mask_type
        self.obj_labels = obj_labels
        assert mask_type in [None, 'bboxes', 'masks']

    def __call__(self, image, target):
        assert target is not None, "target can not be None, it has to include bboxes or masks for cropping based on RoI"
        if self.mask_type == 'bboxes':
            mask = np.zeros(image.shape[:2])
            for key in target.keys():
                if target[key].type == 'bboxes':
                    box = target[key].value
                    for k in range(box.shape[0]):
                        mask[box[k, 1]:box[k, 3] + 1, box[k, 0]:box[k, 2] + 1] = 1
                    break
        elif self.mask_type == 'masks':
            for key in target.keys():
                if target[key].type == 'masks':
                    mask = target[key].value
                    mask = np.sum(mask[..., self.obj_labels], axis=-1)
                    break
        else:
            mask = np.ones(image.shape[:2], dtype=image.dtype)
        crop_loc = self._crop_region(mask > 0.5)  # return crop hmin, hmax, wmin, wmax
        image = image[crop_loc[0]:crop_loc[1], crop_loc[2]:crop_loc[3], :]

        for key, value in target.items():
            if value.type is None or value.type == 'labels':
                continue
            if value.type == 'masks':
                crop = value.value[crop_loc[0]:crop_loc[1], crop_loc[2]:crop_loc[3], :]
            elif value.type == 'bboxes':
                h, w = image.shape[:2]
                crop = value.value
                crop[:, 0] = np.clip(crop[:, 0] - crop_loc[2], 0, w - 1)
                crop[:, 1] = np.clip(crop[:, 1] - crop_loc[0], 0, h - 1)
                crop[:, 2] = np.clip(crop[:, 2] - crop_loc[2], 0, w - 1)
                crop[:, 3] = np.clip(crop[:, 3] - crop_loc[0], 0, h - 1)
                flag = (crop[:, 3] - crop[:, 1] > 1) & (crop[:, 2] - crop[:, 0] > 1)
                crop = crop[flag, :]
            elif value.type == 'points':
                h, w = image.shape[-2:]
                crop = value.value
                crop[:, 0] = np.clip(crop[:, 0] - crop_loc[2], 0, w - 1)
                crop[:, 1] = np.clip(crop[:, 1] - crop_loc[0], 0, h - 1)
                flag = (crop[:, 0] > 0) & (crop[:, 0] < w - 1) & (crop[:, 1] > 1) & (crop[:, 1] < h - 1)
                crop = crop[flag, :]
            value.value = crop
            target[key] = value
        return image, target

    def _crop_region(self, mask):
        bw = mask
        if (self.mask_type is not None) & (random.random() >= self.prob):
            bw = mask == False  ## crop from background
        xl, yl = self.crop_size[0] // 2, self.crop_size[1] // 2
        xh, yh = self.crop_size[0] - xl, self.crop_size[1] - yl
        ind = np.nonzero(bw)
        if len(ind[0]) == 0:
            x, y = bw.shape[0] // 2, bw.shape[1] // 2
        else:
            loc = np.random.randint(low=0, high=len(ind[0]))
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
