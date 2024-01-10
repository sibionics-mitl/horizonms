import torch
from torch import Tensor, eig
from typing import Tuple, Optional, Union
from torchvision.transforms import functional as F
from abc import ABC, abstractmethod
import numpy as np
import random
from .utils import _input_check_value_range_set, _input_get_value_range_set, \
            _gaussian_kernel_size_check_value_range_set, _gaussian_kernel_size_get_value_range_set
from ..builder import TRANSFORMS


__all__ = ["ImageBase", "Uint8ToFloat", "Identity", "Normalizer",
           "Brightness", "Contrast", "Saturation", "Hue", "Sharpness",
           "Posterize", "Solarize", "AutoContrast", "Equalize", "Invert",
           "GaussianBlur", "GaussianNoise", "Lighting",
           "RandomBrightness", "RandomContrast", "RandomSaturation", "RandomHue", "RandomSharpness",
           "RandomPosterize", "RandomSolarize", "RandomAutoContrast", "RandomEqualize", "RandomInvert",
           "RandomGaussianBlur", "RandomGaussianNoise", "RandomLighting"
]


class ImageBase(ABC, torch.nn.Module):
    """Base for image operators implemented by PyTorch.
    """
    @abstractmethod
    def calculate(self, image):
        """conduct transformation for image.

        Args:
            image (Tensor): image data with dimension CxHxW.
        """
        pass

    def forward(self, image, target=None):
        """implement transformation for image, no transformation is required for target.

        Args:
            image (Tensor): image data with dimension CxHxW.
            target (Dict): target data in dictionary format. Default: `None`.
        """
        image = self.calculate(image)
        if target is None:
            return image
        else:
            return image, target


@TRANSFORMS.register_module()
class Uint8ToFloat(ImageBase):
    """Convert data type from uint8 to float in the range of [0, 1].
    """
    def calculate(self, image) -> Tensor:
        return image.float() / 255.0


@TRANSFORMS.register_module()
class Identity(ImageBase):
    """Return the same image value.
    """
    def __init__(self):
        super().__init__()

    def calculate(self, image) -> Tensor:
        return image


@TRANSFORMS.register_module()
class Normalizer(ImageBase):
    """Image normalization.

    Args:
        mode (str): image normalization method. It can be `'zscore'`, `'zero-one'`, `'negative-positive-one'`, or `'customize'`.
            Default: `'zscore'`.
        shift (float | list | tuple): shift value in normalization. This value is required when `mode='customize'`.
            Default: `None`.
        scale (float | list | tuple): normalizer value in normalization. This value is required when `mode='customize'`.
            Default: `None`.
        image_base (bool): if True, the normalization is conducted for all channels; otherwise, the normalization is conducted for each channel.
            Default: `True`.
        epsilon (float): a small number for calculation stability.
            Default: `1e-10`.
    """
    def __init__(self, mode='zscore', shift=None, scale=None, image_base=True, epsilon=1e-10):
        super().__init__()
        assert mode in ['zscore', 'zero-one', 'negative-positive-one', 'customize'], \
            f"mode shoule be in ['zscore', 'zero-one', 'negative-positive-one', 'customize'], but got {mode}."
        self.mode = mode
        if self.mode == 'customize':
            assert shift is not None and scale is not None
            assert isinstance(shift, list) or isinstance(shift, tuple)
            assert isinstance(scale, list) or isinstance(scale, tuple)
            self.shift = torch.tensor(shift)[:, None, None]
            self.scale = torch.tensor(scale)[:, None, None]
        self.image_base = image_base
        self.epsilon = epsilon

    def calculate(self, image) -> Union[Tensor, np.ndarray]:
        image = image.float()
        if self.mode == 'zscore':
            if self.image_base:
                self.shift = image.mean()
                self.scale = image.std()
            else:
                self.shift = image.mean(dim=(-1, -2), keepdim=True)
                self.scale = image.std(dim=(-1, -2), keepdim=True)
        elif self.mode == 'zero-one':
            if self.image_base:
                self.shift = image.min()
                self.scale = image.max() - self.shift
            else:
                self.shift = image.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
                self.scale = image.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0] - self.shift
        elif self.mode == 'negative-positive-one':
            if self.image_base:
                xmin, xmax = image.min(), image.max()
                self.shift = 0.5 * (xmax + xmin)
                self.scale = 0.5 * (xmax - xmin)
            else:
                xmin = image.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
                xmax = image.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
                self.shift = 0.5 * (xmax + xmin)
                self.scale = 0.5 * (xmax - xmin)
        elif self.mode == 'customize':
            if isinstance(image, torch.Tensor):
                self.scale = self.scale.to(device=image.device, dtype=image.dtype, non_blocking=True)
                self.shift = self.shift.to(device=image.device, dtype=image.dtype, non_blocking=True)
        if self.scale.min() < self.epsilon:
            self.scale += self.epsilon

        image = (image - self.shift) / self.scale
        return image

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mode={self.mode}, '
        repr_str += f'image_base={self.image_base}, '
        repr_str += f'epsilon={self.epsilon})'
        return repr_str


@TRANSFORMS.register_module()
class Brightness(ImageBase):
    """Adjust brightness of an image.

    Args:
        brightness_factor (float | tuple[float, float] | list[float, float]): how much to adjust the brightness.
            It can be any non-negative number. 0 gives a black image, 1 gives the original image while 2 increases the brightness by a factor of 2.
            There are three ways for brightness factor as follows:
                - If `brightness_factor` is `float`, then brightness factor is the value.
                - If `brightness_factor` is `tuple[float, float]` (i.e. a ratio range), then brightness factor is randomly selected from the range.
                - If `brightness_factor` is `list[float, ... , float]` (i.e. list of angles), then brightness factor is randomly selected from the list.
    """
    def __init__(self, brightness_factor):
        super().__init__()
        _input_check_value_range_set(brightness_factor)
        self.brightness_factor = brightness_factor

    def calculate(self, image) -> Tensor:
        brightness_factor_sel = _input_get_value_range_set(self.brightness_factor)
        image = F.adjust_brightness(image, brightness_factor_sel)
        return image

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(brightness_factor={self.brightness_factor})'
        return repr_str

@TRANSFORMS.register_module()
class RandomBrightness(ImageBase):
    """Randomly adjust brightness of an image with a predefined probability.

    Args:
        prob (float): probability of brightness adjustment for an image.
        brightness_factor (float | tuple[float, float] | list[float, float]): how much to adjust the brightness.
            It can be any non-negative number. 0 gives a black image, 1 gives the original image while 2 increases the brightness by a factor of 2.
            There are three ways for brightness factor as follows:
                - If `brightness_factor` is `float`, then brightness factor is the value.
                - If `brightness_factor` is `tuple[float, float]` (i.e. a ratio range), then brightness factor is randomly selected from the range.
                - If `brightness_factor` is `list[float, ... , float]` (i.e. list of angles), then brightness factor is randomly selected from the list.
    """
    def __init__(self, prob, brightness_factor):
        super().__init__()
        _input_check_value_range_set(brightness_factor)
        self.brightness_prob = prob
        self.brightness_factor = brightness_factor

    def calculate(self, image) -> Tensor:
        if random.random() < self.brightness_prob:
            brightness_factor_sel = _input_get_value_range_set(self.brightness_factor)
            image = F.adjust_brightness(image, brightness_factor_sel)
        return image

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'brightness_factor={self.brightness_factor})'
        return repr_str


@TRANSFORMS.register_module()
class Contrast(ImageBase):
    """Adjust contrast of an image.

    Args:
        contrast_factor (float | tuple[float, float] | list[float, float]):  
            how much to adjust the contrast. It can be any non-negative number.
            0 gives a solid gray image, 1 gives the original image while 2 increases the contrast by a factor of 2.
            There are three ways for contrast factor as follows:
                - If `contrast_factor` is `float`, then contrast factor is the value.
                - If `contrast_factor` is `tuple[float, float]` (i.e. a ratio range), then contrast factor is randomly selected from the range.
                - If `contrast_factor` is `list[float, ... , float]` (i.e. list of angles), then contrast factor is randomly selected from the list.
    """
    def __init__(self, contrast_factor):
        super().__init__()
        _input_check_value_range_set(contrast_factor)
        self.contrast_factor = contrast_factor

    def calculate(self, image) -> Tensor:
        contrast_factor_sel = _input_get_value_range_set(self.contrast_factor)
        image = F.adjust_contrast(image, contrast_factor_sel)
        return image

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(contrast_factor={self.contrast_factor})'
        return repr_str


@TRANSFORMS.register_module()
class RandomContrast(ImageBase):
    """Randomly adjust contrast of an image with a predefined probability.

    Args:
        prob (float): probability of contrast adjustment for an image.
        contrast_factor (float | tuple[float, float] | list[float, float]): 
            how much to adjust the contrast. It can be any non-negative number.
            0 gives a solid gray image, 1 gives the original image while 2 increases the contrast by a factor of 2.
            There are three ways for contrast factor as follows:
                - If `contrast_factor` is `float`, then contrast factor is the value.
                - If `contrast_factor` is `tuple[float, float]` (i.e. a ratio range), then contrast factor is randomly selected from the range.
                - If `contrast_factor` is `list[float, ... , float]` (i.e. list of angles), then contrast factor is randomly selected from the list.
    """
    def __init__(self, prob, contrast_factor):
        super().__init__()
        _input_check_value_range_set(contrast_factor)
        self.prob = prob
        self.contrast_factor = contrast_factor

    def calculate(self, image) -> Tensor:
        if random.random() < self.prob:
            contrast_factor_sel = _input_get_value_range_set(self.contrast_factor)
            image = F.adjust_contrast(image, contrast_factor_sel)
        return image

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'contrast_factor={self.contrast_factor})'
        return repr_str


@TRANSFORMS.register_module()
class Saturation(ImageBase):  # Color
    """Adjust color saturation of an image.

    Args:
        saturation_factor (float | tuple[float, float] | list[float, float]):  
            how much to adjust the saturation. 0 will give a black and white image, 
            1 will give the original image, while 2 will enhance the saturation by a factor of 2.
            There are three ways for saturation factor as follows:
                - If `saturation_factor` is `float`, then saturation factor is the value.
                - If `saturation_factor` is `tuple[float, float]` (i.e. a ratio range), then saturation factor is randomly selected from the range.
                - If `saturation_factor` is `list[float, ... , float]` (i.e. list of angles), then saturation factor is randomly selected from the list.
    """
    def __init__(self, saturation_factor):
        super().__init__()
        _input_check_value_range_set(saturation_factor)
        self.saturation_factor = saturation_factor

    def calculate(self, image) -> Tensor:
        saturation_factor_sel = _input_get_value_range_set(self.saturation_factor)
        image = F.adjust_saturation(image, saturation_factor_sel)
        return image

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(saturation_factor={self.saturation_factor})'
        return repr_str


@TRANSFORMS.register_module()
class RandomSaturation(ImageBase):  # Color
    """Randomly adjust color saturation of an image with a predefined probability.

    Args:
        prob (float): probability of saturation adjustment for an image.
        saturation_factor (float | tuple[float, float] | list[float, float]): 
            how much to adjust the saturation. 0 will give a black and white image, 
            1 will give the original image, while 2 will enhance the saturation by a factor of 2.
            There are three ways for saturation factor as follows:
                - If `saturation_factor` is `float`, then saturation factor is the value.
                - If `saturation_factor` is `tuple[float, float]` (i.e. a ratio range), then saturation factor is randomly selected from the range.
                - If `saturation_factor` is `list[float, ... , float]` (i.e. list of angles), then saturation factor is randomly selected from the list.
    """
    def __init__(self, prob, saturation_factor):
        super().__init__()
        _input_check_value_range_set(saturation_factor)
        self.prob = prob
        self.saturation_factor = saturation_factor

    def calculate(self, image) -> Tensor:
        if random.random() < self.prob:
            saturation_factor_sel = _input_get_value_range_set(self.saturation_factor)
            image = F.adjust_saturation(image, saturation_factor_sel)
        return image

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'saturation_factor={self.saturation_factor})'
        return repr_str

@TRANSFORMS.register_module()
class Hue(ImageBase):
    """Adjust hue of an image.

    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.

    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.

    See `Hue`_ for more details.

    .. _Hue: https://en.wikipedia.org/wiki/Hue

    Args:
        hue_factor (float | tuple[float, float] | list[float, float]): 
            how much to shift the hue channel. It should be in [-0.5, 0.5]. 
            0 means no shift, thus gives the original image. 
            0.5 and -0.5 give complete reversal of hue channel in HSV space in positive 
            and negative direction respectively, thus both -0.5 and 0.5 will give an image
            with complementary colors.
            There are three ways for hue factor as follows:
                - If `hue_factor` is `float`, then hue factor is the value.
                - If `hue_factor` is `tuple[float, float]` (i.e. a ratio range), then hue factor is randomly selected from the range.
                - If `hue_factor` is `list[float, ... , float]` (i.e. list of angles), then hue factor is randomly selected from the list.
    """
    def __init__(self, hue_factor):
        super().__init__()
        _input_check_value_range_set(hue_factor)
        self.hue_factor = hue_factor

    def calculate(self, image) -> Tensor:
        hue_factor_sel = _input_get_value_range_set(self.hue_factor)
        image = F.adjust_hue(image, hue_factor_sel)
        return image

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(hue_factor={self.hue_factor})'
        return repr_str

@TRANSFORMS.register_module()
class RandomHue(ImageBase):
    """Randomly adjust hue of an image with a predefined probability.

    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.

    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.

    See `Hue`_ for more details.

    .. _Hue: https://en.wikipedia.org/wiki/Hue

    Args:
        prob (float): probability of hue adjustment for an image.
        hue_factor (float | tuple[float, float] | list[float, float]): 
            how much to shift the hue channel. It should be in [-0.5, 0.5]. 
            0 means no shift, thus gives the original image. 
            0.5 and -0.5 give complete reversal of hue channel in HSV space in positive 
            and negative direction respectively, thus both -0.5 and 0.5 will give an image
            with complementary colors.
            There are three ways for hue factor as follows:
                - If `hue_factor` is `float`, then hue factor is the value.
                - If `hue_factor` is `tuple[float, float]` (i.e. a ratio range), then hue factor is randomly selected from the range.
                - If `hue_factor` is `list[float, ... , float]` (i.e. list of angles), then hue factor is randomly selected from the list.
    """
    def __init__(self, prob, hue_factor):
        super().__init__()
        _input_check_value_range_set(hue_factor)
        self.prob = prob
        self.hue_factor = hue_factor

    def calculate(self, image) -> Tensor:
        if random.random() < self.prob:
            hue_factor_sel = _input_get_value_range_set(self.hue_factor)
            image = F.adjust_hue(image, hue_factor_sel)
        return image

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'hue_factor={self.hue_factor})'
        return repr_str

@TRANSFORMS.register_module()
class Sharpness(ImageBase):
    """Adjust the sharpness of an image.

    Args:
        sharpness_factor (float | tuple[float, float] | list[float, float]): 
            how much to adjust the sharpness. It can be any non-negative number. 
            0 gives a blurred image, 1 gives the original image while 2 increases 
            the sharpness by a factor of 2.
            There are three ways for sharpness factor as follows:
                - If `sharpness_factor` is `float`, then sharpness factor is the value.
                - If `sharpness_factor` is `tuple[float, float]` (i.e. a ratio range), then sharpness factor is randomly selected from the range.
                - If `sharpness_factor` is `list[float, ... , float]` (i.e. list of angles), then sharpness factor is randomly selected from the list.
    """
    def __init__(self, sharpness_factor):
        super().__init__()
        _input_check_value_range_set(sharpness_factor)
        self.sharpness_factor = sharpness_factor

    def calculate(self, image) -> Tensor:
        sharpness_factor_sel = _input_get_value_range_set(self.sharpness_factor)
        image = F.adjust_sharpness(image, sharpness_factor_sel)
        return image

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(sharpness_factor={self.sharpness_factor})'
        return repr_str

@TRANSFORMS.register_module()
class RandomSharpness(ImageBase):
    """Randomly adjust the sharpness of an image with a predefined probability.

    Args:
        prob (float): probability of sharpness adjustment for an image.
        sharpness_factor (float | tuple[float, float] | list[float, float]): 
            how much to adjust the sharpness. It can be any non-negative number. 
            0 gives a blurred image, 1 gives the original image while 2 increases 
            the sharpness by a factor of 2.
            There are three ways for sharpness factor as follows:
                - If `sharpness_factor` is `float`, then sharpness factor is the value.
                - If `sharpness_factor` is `tuple[float, float]` (i.e. a ratio range), then sharpness factor is randomly selected from the range.
                - If `sharpness_factor` is `list[float, ... , float]` (i.e. list of angles), then sharpness factor is randomly selected from the list.
    """
    def __init__(self, prob, sharpness_factor):
        super().__init__()
        _input_check_value_range_set(sharpness_factor)
        self.prob = prob
        self.sharpness_factor = sharpness_factor

    def calculate(self, image) -> Tensor:
        if random.random() < self.prob:
            sharpness_factor_sel = _input_get_value_range_set(self.sharpness_factor)
            image = F.adjust_sharpness(image, sharpness_factor_sel)
        return image

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'sharpness_factor={self.sharpness_factor})'
        return repr_str

@TRANSFORMS.register_module()
class Posterize(ImageBase):
    """Posterize an image by reducing the number of bits for each color channel.

    Args:
        posterize_bins (float | tuple[float, float] | list[float, float]): number of bits to keep for each channel.
        There are three ways for bins as follows:
                - If `posterize_bins` is `float`, then bins are the value.
                - If `posterize_bins` is `tuple[float, float]` (i.e. a ratio range), then bins are randomly selected from the range.
                - If `posterize_bins` is `list[float, ... , float]` (i.e. list of angles), then bins are randomly selected from the list.
    """
    def __init__(self, posterize_bins):
        super().__init__()
        _input_check_value_range_set(posterize_bins, dtype='int')
        self.posterize_bins = posterize_bins

    def calculate(self, image) -> Tensor:
        posterize_bins_sel = _input_get_value_range_set(self.posterize_bins, dtype='int')
        if image.is_floating_point():
            image = (image * 255).type(dtype=torch.uint8)
            image = F.posterize(image, posterize_bins_sel)
            image = image.float() / 255.0
        else:
            image = F.posterize(image, posterize_bins_sel)
        return image

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(posterize_bins={self.posterize_bins})'
        return repr_str

@TRANSFORMS.register_module()
class RandomPosterize(ImageBase):
    """Randomly posterize an image by reducing the number of bits for each color channel.

    Args:
        prob (float): probability of posterizing an image.
        posterize_bins (float | tuple[float, float] | list[float, float]): number of bits to keep for each channel.
        There are three ways for bins as follows:
                - If `posterize_bins` is `float`, then bins are the value.
                - If `posterize_bins` is `tuple[float, float]` (i.e. a ratio range), then bins are randomly selected from the range.
                - If `posterize_bins` is `list[float, ... , float]` (i.e. list of angles), then bins are randomly selected from the list.
    """
    def __init__(self, prob, posterize_bins):
        super().__init__()
        _input_check_value_range_set(posterize_bins, dtype='int')
        self.prob = prob
        self.posterize_bins = posterize_bins

    def calculate(self, image) -> Tensor:
        if random.random() < self.prob:
            posterize_bins_sel = _input_get_value_range_set(self.posterize_bins, dtype='int')
            if image.is_floating_point():
                image = (image * 255).type(dtype=torch.uint8)
                image = F.posterize(image, posterize_bins_sel)
                image = image.float() / 255.0
            else:
                image = F.posterize(image, posterize_bins_sel)
        return image

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'posterize_bins={self.posterize_bins})'
        return repr_str

@TRANSFORMS.register_module()
class Solarize(ImageBase):
    """Solarize an RGB/grayscale image by inverting all pixel values above a threshold.

    Args:
        solarize_threshold (float | tuple[float, float] | list[float, float]): threshold for solarizing.
            It is in range between 0 and 1. All pixels equal or above this value are inverted.
            There are three ways for solarizing threshold as follows:
                - If `solarize_threshold` is `float`, then solarizing threshold is the value.
                - If `solarize_threshold` is `tuple[float, float]` (i.e. a ratio range), then solarizing threshold is randomly selected from the range.
                - If `solarize_threshold` is `list[float, ... , float]` (i.e. list of angles), then solarizing threshold is randomly selected from the list.
    """
    def __init__(self, solarize_threshold):
        super().__init__()
        _input_check_value_range_set(solarize_threshold)
        self.solarize_threshold = solarize_threshold

    def calculate(self, image) -> Tensor:
        solarize_threshold_sel = _input_get_value_range_set(self.solarize_threshold)
        if not image.is_floating_point():
            solarize_threshold_sel = solarize_threshold_sel * 255
        image = F.solarize(image, solarize_threshold_sel)
        return image

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(solarize_threshold={self.solarize_threshold})'
        return repr_str

@TRANSFORMS.register_module()
class RandomSolarize(ImageBase):
    """Randomly solarize an RGB/grayscale image by inverting all pixel values above a threshold.

    Args:
        prob (float): probability of solarizing an image.
        solarize_threshold (float | tuple[float, float] | list[float, float]): threshold for solarizing.
            It is in range between 0 and 1. All pixels equal or above this value are inverted.
            There are three ways for solarizing threshold as follows:
                - If `solarize_threshold` is `float`, then solarizing threshold is the value.
                - If `solarize_threshold` is `tuple[float, float]` (i.e. a ratio range), then solarizing threshold is randomly selected from the range.
                - If `solarize_threshold` is `list[float, ... , float]` (i.e. list of angles), then solarizing threshold is randomly selected from the list.
    """
    def __init__(self, prob, solarize_threshold=None):
        super().__init__()
        _input_check_value_range_set(solarize_threshold)
        self.prob = prob
        self.solarize_threshold = solarize_threshold

    def calculate(self, image) -> Tensor:
        if random.random() < self.prob:
            solarize_threshold_sel = _input_get_value_range_set(self.solarize_threshold)
            if not image.is_floating_point():
                solarize_threshold_sel = solarize_threshold_sel * 255
            image = F.solarize(image, solarize_threshold_sel)
        return image

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'solarize_threshold={self.solarize_threshold})'
        return repr_str

@TRANSFORMS.register_module()
class AutoContrast(ImageBase):
    """Maximize contrast of an image by remapping its
    pixels per channel so that the lowest becomes black and the lightest
    becomes white.
    """
    def __init__(self):
        super().__init__()

    def calculate(self, image) -> Tensor:
        image = F.autocontrast(image)
        return image

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str

@TRANSFORMS.register_module()
class RandomAutoContrast(ImageBase):
    """Randomly maximize contrast of an image by remapping its
    pixels per channel so that the lowest becomes black and the lightest
    becomes white.

    Args:
        prob (float): robability of auto contrast adjustment for an image.
    """
    def __init__(self, prob):
        super().__init__()
        self.prob = prob

    def calculate(self, image) -> Tensor:
        if random.random() < self.prob:
            image = F.autocontrast(image)
        return image

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob})'
        return repr_str

@TRANSFORMS.register_module()
class Equalize(ImageBase):
    """Equalize the histogram of an image by applying
    a non-linear mapping to the input in order to create a uniform
    distribution of grayscale values in the output.
    """
    def __init__(self):
        super().__init__()

    def calculate(self, image) -> Tensor:
        if image.is_floating_point():
            image = (image * 255).type(dtype=torch.uint8)
            image = F.equalize(image)  # F.equalize的内部实现是在0-255数据范围内,如果是0-1范围,效果不正确
            image = image.float() / 255.0
        else:
            image = F.equalize(image)
        return image

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str

@TRANSFORMS.register_module()
class RandomEqualize(ImageBase):
    """Randomly equalize the histogram of an image by applying
    a non-linear mapping to the input in order to create a uniform
    distribution of grayscale values in the output.

    Args:
        prob (float): probability of equalization for an image.
    """
    def __init__(self, prob):
        super().__init__()
        self.prob = prob

    def calculate(self, image) -> Tensor:
        if random.random() < self.prob:
            if image.is_floating_point():
                image = (image * 255).type(dtype=torch.uint8)
                image = F.equalize(image)  # F.equalize的内部实现是在0-255数据范围内,如果是0-1范围,效果不正确
                image = image.float() / 255.0
            else:
                image = F.equalize(image)
        return image

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob})'
        return repr_str

@TRANSFORMS.register_module()
class Invert(ImageBase):
    """Invert the colors of an RGB/grayscale image.
    """
    def __init__(self):
        super().__init__()

    def calculate(self, image) -> Tensor:
        image = F.invert(image)
        return image

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@TRANSFORMS.register_module()
class RandomInvert(ImageBase):
    """Randomly invert the colors of an RGB/grayscale image with a predefined probability.

    Args:
        prob (float): probability of inverting for an image.
    """
    def __init__(self, prob):
        super().__init__()
        self.prob = prob

    def calculate(self, image) -> Tensor:
        if random.random() < self.prob:
            image = F.invert(image)
        return image

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob})'
        return repr_str


@TRANSFORMS.register_module()
class GaussianBlur(ImageBase):
    """Perform Gaussian blurring on the image by given kernel.

    Args:
        sigma (float | tuple[float, float] | list[float, float]): standard deviation of Gaussian kernel.
           There are three ways for standard deviation of Gaussian kernel as follows:
                - If `sigma` is `float`, then standard deviation of Gaussian kernel is the value.
                - If `sigma` is `tuple[float, float]` (i.e. a ratio range), then standard deviation of Gaussian kernel is randomly selected from the range.
                - If `sigma` is `list[float, ... , float]` (i.e. list of angles), then standard deviation of Gaussian kernel is randomly selected from the list.
        kernel_size (float | tuple[float, float] | list[float, float]): size of Gaussian kernel.
            There are three ways for size of Gaussian kernel as follows:
                - If `kernel_size` is `float`, then size of Gaussian kernel is the value.
                - If `kernel_size` is `tuple[float, float]` (i.e. a ratio range), then size of Gaussian kernel is randomly selected from the range.
                - If `kernel_size` is `list[float, ... , float]` (i.e. list of angles), then size of Gaussian kernel is randomly selected from the list.
    """
    def __init__(self, sigma, kernel_size):
        super().__init__()
        _input_check_value_range_set(sigma)
        _gaussian_kernel_size_check_value_range_set(kernel_size)
        self.sigma = sigma
        self.kernel_size = kernel_size

    def calculate(self, image) -> Tensor:
        sigma_sel = _input_get_value_range_set(self.sigma)
        kernel_size_sel = _gaussian_kernel_size_get_value_range_set(self.kernel_size)
        image = F.gaussian_blur(image, (kernel_size_sel, kernel_size_sel), [sigma_sel, sigma_sel])
        return image

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(sigma={self.sigma}, '
        repr_str += f'kernel_size={self.kernel_size})'
        return repr_str


@TRANSFORMS.register_module()
class RandomGaussianBlur(ImageBase):
    """Randomly perform Gaussian blurring on the image by given kernel with a predefined probability.

    Args:
        prob (float): probability of Gaussian blurring for an image.
        sigma (float | tuple[float, float] | list[float, float]): standard deviation of Gaussian kernel.
           There are three ways for standard deviation of Gaussian kernel as follows:
                - If `sigma` is `float`, then standard deviation of Gaussian kernel is the value.
                - If `sigma` is `tuple[float, float]` (i.e. a ratio range), then standard deviation of Gaussian kernel is randomly selected from the range.
                - If `sigma` is `list[float, ... , float]` (i.e. list of angles), then standard deviation of Gaussian kernel is randomly selected from the list.
        kernel_size (float | tuple[float, float] | list[float, float]): size of Gaussian kernel.
            There are three ways for size of Gaussian kernel as follows:
                - If `kernel_size` is `float`, then size of Gaussian kernel is the value.
                - If `kernel_size` is `tuple[float, float]` (i.e. a ratio range), then size of Gaussian kernel is randomly selected from the range.
                - If `kernel_size` is `list[float, ... , float]` (i.e. list of angles), then size of Gaussian kernel is randomly selected from the list.
    """
    def __init__(self, prob, sigma, kernel_size):
        super().__init__()
        _input_check_value_range_set(sigma)
        _gaussian_kernel_size_check_value_range_set(kernel_size)
        self.prob = prob
        self.sigma = sigma
        self.kernel_size = kernel_size

    def calculate(self, image) -> Tensor:
        if random.random() < self.prob:
            sigma_sel = _input_get_value_range_set(self.sigma)
            kernel_size_sel = _gaussian_kernel_size_get_value_range_set(self.kernel_size)
            image = F.gaussian_blur(image, (kernel_size_sel, kernel_size_sel), [sigma_sel, sigma_sel])
        return image

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'sigma={self.sigma}, '
        repr_str += f'kernel_size={self.kernel_size})'
        return repr_str


def _gaussian_noise(image, std, mean):
    dtype = image.dtype
    bound = 1.0 if image.is_floating_point() else 255.0
    if not image.is_floating_point():
        std = std * 255.0
        mean = mean * 255.0
    noise = torch.randn(image.size(), device=image.device) * std + mean
    image = (image + noise).clamp(0, bound).type(dtype)
    # if len(image.shape) == 2:
    #     image = image[..., None]
    return image

@TRANSFORMS.register_module()
class GaussianNoise(ImageBase):
    """Perform Gaussian noise on the image by given std.

    Args:
        std (float | tuple[float, float] | list[float, float]): standard deviation of Gaussian kernel.
            There are three ways for standard deviation of Gaussian kernel as follows:
                - If `std` is `float`, then standard deviation of Gaussian kernel is the value.
                - If `std` is `tuple[float, float]` (i.e. a ratio range), then standard deviation of Gaussian kernel is randomly selected from the range.
                - If `std` is `list[float, ... , float]` (i.e. list of angles), then standard deviation of Gaussian kernel is randomly selected from the list.
        mean (float | tuple[float, float] | list[float, float]): mean of Gaussian kernel.
            There are three ways for mean of Gaussian kernel as follows:
                - If `mean` is `float`, then mean of Gaussian kernel is the value.
                - If `mean` is `tuple[float, float]` (i.e. a ratio range), then mean of Gaussian kernel is randomly selected from the range.
                - If `mean` is `list[float, ... , float]` (i.e. list of angles), then mean of Gaussian kernel is randomly selected from the list.
            Default: `0.0`.
    """
    def __init__(self, std, mean=0.0):
        super().__init__()
        _input_check_value_range_set(std)
        _input_check_value_range_set(mean)
        self.std = std
        self.mean = mean

    def calculate(self, image) -> Tensor:
        std_sel = _input_get_value_range_set(self.std)
        mean_sel = _input_get_value_range_set(self.mean)
        image = _gaussian_noise(image, std_sel, mean_sel)
        return image

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(std={self.std}, '
        repr_str += f'mean={self.mean})'
        return repr_str


@TRANSFORMS.register_module()
class RandomGaussianNoise(ImageBase):
    """Randomly perform Gaussian noise on the image by given std with a predefined probability.

    Args:
        prob ( float): probability of adding Gaussian noise for an image.
        std (float | tuple[float, float] | list[float, float]): standard deviation of Gaussian kernel.
            There are three ways for standard deviation of Gaussian kernel as follows:
                - If `std` is `float`, then standard deviation of Gaussian kernel is the value.
                - If `std` is `tuple[float, float]` (i.e. a ratio range), then standard deviation of Gaussian kernel is randomly selected from the range.
                - If `std` is `list[float, ... , float]` (i.e. list of angles), then standard deviation of Gaussian kernel is randomly selected from the list.
        mean (float | tuple[float, float] | list[float, float]): mean of Gaussian kernel.
            There are three ways for mean of Gaussian kernel as follows:
                - If `mean` is `float`, then mean of Gaussian kernel is the value.
                - If `mean` is `tuple[float, float]` (i.e. a ratio range), then mean of Gaussian kernel is randomly selected from the range.
                - If `mean` is `list[float, ... , float]` (i.e. list of angles), then mean of Gaussian kernel is randomly selected from the list.
            Default: `0.0`.
    """
    def __init__(self, prob, std, mean=0.0):
        super().__init__()
        _input_check_value_range_set(std)
        _input_check_value_range_set(mean)
        self.prob = prob
        self.std = std
        self.mean = mean

    def calculate(self, image) -> Tensor:
        if random.random() < self.prob:
            std_sel = _input_get_value_range_set(self.std)
            mean_sel = _input_get_value_range_set(self.mean)
            image = _gaussian_noise(image, std_sel, mean_sel)
        return image

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'std={self.std}, '
        repr_str += f'mean={self.mean})'
        return repr_str


def _lighting(image, alphastd, eigvec, eigval):
    if alphastd == 0:
        return image
    # alpha = torch.tensor(np.array([0.4638, 0.2112, 0.1410]))
    # alpha = image.new().resize_(3).normal_(0, alphastd)  # 差异来自alpha的生成,torch的随机数和random的随机数不一致
    alpha = torch.tensor(np.array([random.normalvariate(0, alphastd, ) for i in range(3)]),
                         device=image.device)  # 为了和cv版生成随机数一致

    rgb = eigvec.type_as(image).clone() \
        .mul(alpha.view(1, 3).expand(3, 3)) \
        .mul(eigval.view(1, 3).expand(3, 3)) \
        .sum(1).squeeze()
    image = image.add(rgb.view(3, 1, 1).expand_as(image))
    return image


@TRANSFORMS.register_module()
class Lighting(ImageBase):
    """Lighting noise (AlexNet - style PCA - based noise)
    codes from: https://github.com/automl/trivialaugment/blob/3bfd06552336244b23b357b2c973859500328fbb/TrivialAugment/augmentations.py
    """
    def __init__(self, alphastd, eigval, eigvec):
        super().__init__()
        _input_check_value_range_set(alphastd)
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def calculate(self, image) -> Tensor:
        alphastd_sel = _input_get_value_range_set(self.alphastd)
        if image.is_floating_point():
            image = _lighting(image, alphastd_sel, self.eigvec, self.eigval)  # 差异来自alpha的生成,torch的随机数和random的随机数不一致
        else:
            image = image.to(torch.float32) / 255.
            image = _lighting(image, alphastd_sel, self.eigvec, self.eigval)  # 差异来自alpha的生成,torch的随机数和random的随机数不一致
            image = torch.clamp(image * 255., 0, 255)
            image = image.to(torch.uint8)
        return image

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(alphastd={self.alphastd}, '
        repr_str += f'eigval={self.eigval}, '
        repr_str += f'eigvec={self.eigvec})'
        return repr_str


@TRANSFORMS.register_module()
class RandomLighting(ImageBase):
    """Lighting noise (AlexNet - style PCA - based noise)
    codes from: https://github.com/automl/trivialaugment/blob/3bfd06552336244b23b357b2c973859500328fbb/TrivialAugment/augmentations.py

    Args:
        prob (float): probability of image Lighting.
    """
    def __init__(self, prob):
        super().__init__()
        self.prob = prob

    def calculate(self, image) -> Tensor:
        if np.random.random() < self.prob:
            alphastd = torch.tensor(np.random.uniform(0, 1), device=image.device)  ## 为了和cv版生成随机数一致
            eigval = torch.tensor(np.random.uniform(0, 1, (image.shape[0])), device=image.device)
            eigvec = torch.tensor(np.random.uniform(0, 1, (image.shape[0], image.shape[0])), device=image.device)

            if image.is_floating_point():
                image = _lighting(image, alphastd, eigvec, eigval)  # 差异来自alpha的生成,torch的随机数和random的随机数不一致
            else:
                image = image.to(torch.float32) / 255.
                image = _lighting(image, alphastd, eigvec, eigval)  # 差异来自alpha的生成,torch的随机数和random的随机数不一致
                image = torch.clamp(image * 255., 0, 255)
                image = image.to(torch.uint8)
        return image

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob})'
        return repr_str
