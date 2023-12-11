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


__all__ = ("ImageBase", "Uint8ToFloat", "Identity", "Normalizer",
           "Brightness", "Contrast", "Saturation", "Hue", "Sharpness",
           "Posterize", "Solarize", "AutoContrast", "Equalize", "Invert",
           "GaussianBlur", "GaussianNoise", "Lighting",
           "RandomBrightness", "RandomContrast", "RandomSaturation", "RandomHue", "RandomSharpness",
           "RandomPosterize", "RandomSolarize", "RandomAutoContrast", "RandomEqualize", "RandomInvert",
           "RandomGaussianBlur", "RandomGaussianNoise", "RandomLighting"
           )


class ImageBase(ABC, torch.nn.Module):

    @abstractmethod
    def calculate(self, image):
        pass

    def forward(self, image, target=None):
        image = self.calculate(image)
        if target is None:
            return image
        else:
            return image, target


@TRANSFORMS.register_module()
class Uint8ToFloat(ImageBase):
    def calculate(self, image) -> Tensor:
        return image.float() / 255.0


@TRANSFORMS.register_module()
class Identity(ImageBase):
    def __init__(self):
        super().__init__()

    def calculate(self, image) -> Tensor:
        return image


@TRANSFORMS.register_module()
class Normalizer(ImageBase):
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
    """Adjust brightness of an image

    Args:
        brightness_factor (float  | tuple[float, float] |  list[float, float] )
                float : How much to adjust the brightness.
                        Can be any non-negative number. 0 gives a black image, 1 gives
                        the original image while 2 increases the brightness by a factor of 2.
                tuple[float, float]: Range of the ratio of brightness .
                list[float, ... , float] : list of the ratio of brightness to be randomly chosen.
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
    """Adjust brightness of an image

    Args:
        prob (float): Probability of image adjusting brightness.
        brightness_factor (float  | tuple[float, float] |  list[float, float] )
                float : How much to adjust the brightness.
                        Can be any non-negative number. 0 gives a black image, 1 gives
                        the original image while 2 increases the brightness by a factor of 2.
                tuple[float, float]: Range of the ratio of brightness .
                list[float, ... , float] : list of the ratio of brightness to be randomly chosen.
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
        contrast_factor (float  | tuple[float, float] |  list[float, float] )
                float : How much to adjust the contrast. Can be any
                    non-negative number. 0 gives a solid gray image, 1 gives the
                    original image while 2 increases the contrast by a factor of 2.
                tuple[float, float]:  Range of the ratio of contrast.
                list[float, ... , float] : list of the ratio of contrast to be randomly chosen.
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
    """Adjust contrast of an image.

    Args:
        prob (float): Probability of image adjusting contrast.
        contrast_factor (float  | tuple[float, float] |  list[float, float] )
                float : How much to adjust the contrast. Can be any
                    non-negative number. 0 gives a solid gray image, 1 gives the
                    original image while 2 increases the contrast by a factor of 2.
                tuple[float, float]:  Range of the ratio of contrast.
                list[float, ... , float] : list of the ratio of contrast to be randomly chosen.
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
        saturation_factor (float  | tuple[float, float] |  list[float, float] )
                float:  How much to adjust the saturation. 0 will
                    give a black and white image, 1 will give the original image while
                    2 will enhance the saturation by a factor of 2.
                tuple[float, float]: Range of the ratio of saturation.
                list[float, ... , float] : list of the ratio of saturation to be randomly chosen.
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
    """Adjust color saturation of an image.

    Args:
        prob (float):Probability of image adjusting sharpness.
        saturation_factor (float  | tuple[float, float] |  list[float, float] )
                float:  How much to adjust the saturation. 0 will
                    give a black and white image, 1 will give the original image while
                    2 will enhance the saturation by a factor of 2.
                tuple[float, float]: Range of the ratio of saturation.
                list[float, ... , float] : list of the ratio of saturation to be randomly chosen.
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
        hue_factor  (float  | tuple[float, float] |  list[float, float] ) Range between -0.5 and 0.5:
            - float:  How much to shift the hue channel. Should be in
                    [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
                    HSV space in positive and negative direction respectively.
                    0 means no shift. Therefore, both -0.5 and 0.5 will give an image
                    with complementary colors while 0 gives the original image.
            - tuple[float, float]: Range of the ratio of hue.
            - list[float, ... , float] : list of the ratio of hue to be randomly chosen.
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
    """Adjust hue of an image.
    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.

    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.

    See `Hue`_ for more details.

    .. _Hue: https://en.wikipedia.org/wiki/Hue

    Args:
        prob (float): Probability of image adjusting hue.
        hue_factor  (float  | tuple[float, float] |  list[float, float] ) Range between -0.5 and 0.5:
            - float:  How much to shift the hue channel. Should be in
                    [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
                    HSV space in positive and negative direction respectively.
                    0 means no shift. Therefore, both -0.5 and 0.5 will give an image
                    with complementary colors while 0 gives the original image.
            - tuple[float, float]: Range of the ratio of hue.
            - list[float, ... , float] : list of the ratio of hue to be randomly chosen.
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
        sharpness_factor (float  | tuple[float, float] |  list[float, float] ):
            - float:  How much to adjust the sharpness. Can be
                    any non-negative number. 0 gives a blurred image, 1 gives the
                    original image while 2 increases the sharpness by a factor of 2.
            - tuple[float, float]: Range of the ratio of sharpness.
            - list[float, ... , float] : list of the ratio of sharpness to be randomly chosen.
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
    """Adjust the sharpness of an image.

    Args:
        prob (float):  Probability of image adjusting sharpness.
        sharpness_factor (float  | tuple[float, float] |  list[float, float] ):
            - float:  How much to adjust the sharpness. Can be
                    any non-negative number. 0 gives a blurred image, 1 gives the
                    original image while 2 increases the sharpness by a factor of 2.
            - tuple[float, float]: Range of the ratio of sharpness.
            - list[float, ... , float] : list of the ratio of sharpness to be randomly chosen.
    """

    def __init__(self, prob, sharpness_factor=None):
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
        posterize_bins  (int  | tuple[int, int] |  list[int, int] ) Range between 0 and 8:
            - int: The number of bits to keep for each channel (0-8).
            - tuple[int, int]: Range of the number of bits.
            - list[int, ... , int] : list of the number to be randomly chosen.
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
    """Posterize an image by reducing the number of bits for each color channel.

    Args:
        prob (float):  Probability of image adjusting posterize.
        posterize_bins  (int  | tuple[int, int] |  list[int, int] ) Range between 0 and 8:
            - int: The number of bits to keep for each channel (0-8).
            - tuple[int, int]: Range of the number of bits.
            - list[int, ... , int] : list of the number to be randomly chosen.
    """

    def __init__(self, prob, posterize_bins=None):
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
        solarize_threshold (float  | tuple[float, float] |  list[float, float] ) Range between 0 and 1:
            - float: All pixels equal or above this value are inverted.
            - tuple[float, float]: Range of values to be randomly chosen.
            - list[float, ... , float] : list of values to be randomly chosen.
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
    """Solarize an RGB/grayscale image by inverting all pixel values above a threshold.

    Args:
        prob (float):  Probability of image adjusting solarize.
        solarize_threshold (float  | tuple[float, float] |  list[float, float] ) Range between 0 and 1:
            - float: All pixels equal or above this value are inverted.
            - tuple[float, float]: Range of values to be randomly chosen.
            - list[float, ... , float] : list of values to be randomly chosen.
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
    """Maximize contrast of an image by remapping its
    pixels per channel so that the lowest becomes black and the lightest
    becomes white.

    Args:
        contrast_prob  (float):  Probability of image adjusting contrast.
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
    """Equalize the histogram of an image by applying
    a non-linear mapping to the input in order to create a uniform
    distribution of grayscale values in the output.

    Args:
        prob  (float):  Probability of image adjusting equalize.
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
    """Invert the colors of an RGB/grayscale image.

    Args:
        prob (float):  Probability of image adjusting invert.
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
    """Performs Gaussian blurring on the image by given kernel.

    Args:
        sigma (float  | tuple[float, float] |  list[float, float] ):
            - float: Gaussian kernel standard deviation.
            - tuple[float, float]: Range of the standard deviation.
            - list[float, float]: list of the standard deviation to be randomly chosen.
        kernel_size (int  | tuple[int, int] |  list[int, int] ):
            - int: Gaussian kernel size.
            - tuple[int, int]: Range of the Gaussian kernel size.
            - list[int, int]: list of the  Gaussian kernel size to be randomly chosen.
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
    """Performs Gaussian blurring on the image by given kernel.
    Args:
        blur_prob (float):  Probability of image adjusting Gaussian blurring.
        sigma (float  | tuple[float, float] |  list[float, float] ):
            - float: Gaussian kernel standard deviation.
            - tuple[float, float]: Range of the standard deviation.
            - list[float, float]: list of the standard deviation to be randomly chosen.
        kernel_size (int  | tuple[int, int] |  list[int, int] ):
            - int: Gaussian kernel size.
            - tuple[int, int]: Range of the Gaussian kernel size.
            - list[int, int]: list of the  Gaussian kernel size to be randomly chosen.
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
    """Performs Gaussian noise on the image by given std.

    Args:
        std (float  | tuple[float, float] |  list[float, float] ):
            - float: Gaussian kernel standard deviation.
            - tuple[float, float]: Range of the standard deviation.
            - list[float, float]: list of the standard deviation to be randomly chosen.
        mean (float  | tuple[float, float] |  list[float, float] ):
            - float: Gaussian kernel mean.
            - tuple[float, float]: Range of the Gaussian kernel mean.
            - list[float, float]: list of the Gaussian kernel mean to be randomly chosen.
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
    """Performs Gaussian noise on the image by given std.

    Args:
        noise_prob ( float):  Probability of image adjusting Gaussian noise.
        std (float  | tuple[float, float] |  list[float, float] ):
            - float: Gaussian kernel standard deviation.
            - tuple[float, float]: Range of the standard deviation.
            - list[float, float]: list of the standard deviation to be randomly chosen.
        mean (float  | tuple[float, float] |  list[float, float] ):
            - float: Gaussian kernel mean.
            - tuple[float, float]: Range of the Gaussian kernel mean.
            - list[float, float]: list of the Gaussian kernel mean to be randomly chosen.
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
    """Lighting noise(AlexNet - style PCA - based noise)
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
    """Lighting noise(AlexNet - style PCA - based noise)
    codes from: https://github.com/automl/trivialaugment/blob/3bfd06552336244b23b357b2c973859500328fbb/TrivialAugment/augmentations.py

    Args:
        prob (float):  Probability of image adjusting Lighting noise.
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
