from re import RegexFlag
import cv2
import random
import numpy as np
from typing import List
from abc import ABC, abstractmethod
from .utils import _input_check_value_range_set, _input_get_value_range_set, \
            _gaussian_kernel_size_check_value_range_set, _gaussian_kernel_size_get_value_range_set
from ..builder import TRANSFORMS


__all__ = ("CVImageBase", "CVUint8ToFloat", "CVIdentity", "CVCvtColor", "CVNormalizer",
           "CVBrightness", "CVContrast", "CVSaturation", "CVHue", "CVSharpness",
           "CVPosterize", "CVSolarize", "CVAutoContrast", "CVEqualize", "CVInvert",
           "CVGaussianBlur", "CVGaussianNoise", "CVLighting",
           "CVRandomBrightness", "CVRandomContrast", "CVRandomSaturation", "CVRandomHue", "CVRandomSharpness",
           "CVRandomPosterize", "CVRandomSolarize", "CVRandomAutoContrast", "CVRandomEqualize", "CVRandomInvert",
           "CVRandomGaussianBlur", "CVRandomGaussianNoise", "CVRandomLighting", "CVRandomBlur",
           )


IMG_ROW_AXIS = 0
IMG_COL_AXIS = 1
IMG_CHANNEL_AXIS = 2


def _max_value_cv(dtype: np.dtype) -> int:
    if dtype == np.uint8:
        return 255
    elif dtype == np.int8:
        return 127
    elif dtype == np.int16:
        return 32767
    elif dtype == np.int32:
        return 2147483647
    elif dtype == np.int64:
        return 9223372036854775807
    else:
        # This is only here for completeness. This value is implicitly assumed in a lot of places so changing it is not
        # easy.
        return 1


def _is_floating_point_cv(dtype: np.dtype) -> bool:
    if dtype in [np.float16, np.float32, np.float64]:
        return True
    else:
        return False


def _assert_channels_cv(img, permitted) -> None:
    c = _get_dimensions_cv(img)[-1]
    if c not in permitted:
        raise TypeError(f"Input image tensor permitted channel values are {permitted}, but found {c}")


def _get_dimensions_cv(img) -> List[int]:
    channels = 1 if len(img.shape) == 2 else img.shape[IMG_CHANNEL_AXIS]
    height, width = img.shape[:2]
    return [height, width, channels]


def _blend_cv(img1, img2, ratio: float):
    ratio = float(ratio)
    bound = _max_value_cv(img1.dtype)
    return np.clip((ratio * img1 + (1.0 - ratio) * img2), 0, bound).astype(img1.dtype)


def _blurred_degenerate_image_cv(img):
    dtype = img.dtype if _is_floating_point_cv(img.dtype) else np.float32

    kernel = np.ones((3, 3), dtype=dtype)
    kernel[1, 1] = 5.0
    kernel /= np.sum(kernel)

    result_tmp = cv2.filter2D(img, cv2.CV_32F, kernel)
    if len(result_tmp.shape) == 2:
        result_tmp = result_tmp[..., None]
    return result_tmp


def _rgb_to_grayscale_cv(img):
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    l_img = (0.2989 * r + 0.587 * g + 0.114 * b).astype(img.dtype)
    return l_img[..., None]


class CVImageBase(ABC):

    @abstractmethod
    def calculate(self, image):
        pass

    def __call__(self, image, target=None):
        image = self.calculate(image)
        if target is None:
            return image
        else:
            return image, target


@TRANSFORMS.register_module()
class CVUint8ToFloat(CVImageBase):
    def calculate(self, image):
        return image.astype(float) / 255.0


@TRANSFORMS.register_module()
class CVIdentity(CVImageBase):
    def __init__(self):
        super().__init__()

    def calculate(self, image):
        return image


@TRANSFORMS.register_module()
class CVCvtColor(CVImageBase):
    def __init__(self, code):
        self.code = eval(code)

    def calculate(self, image):
        image = cv2.cvtColor(image, self.code)
        return image


@TRANSFORMS.register_module()
class CVNormalizer(CVImageBase):
    def __init__(self, mode='zscore', shift=None, scale=None, image_base=True, epsilon=1e-10):
        super().__init__()
        assert mode in ['zscore', 'zero-one', 'negative-positive-one', 'customize'], \
            f"mode shoule be in ['zscore', 'zero-one', 'negative-positive-one', 'customize'], but got {mode}."
        self.mode = mode
        if self.mode == 'customize':
            assert shift is not None and scale is not None
            assert isinstance(shift, list) or isinstance(shift, tuple)
            assert isinstance(scale, list) or isinstance(scale, tuple)
            self.shift = np.array(shift)[None, None, :].astype(np.float32)
            self.scale = np.array(scale)[None, None, :].astype(np.float32)
        self.image_base = image_base
        self.epsilon = epsilon

    def calculate(self, image):
        image = image.astype(np.float32)
        if self.mode == 'zscore':
            if self.image_base:
                self.shift = image.mean()
                self.scale = image.std()
            else:
                self.shift = image.mean(axis=(0, 1), keepdims=True)
                self.scale = image.std(axis=(0, 1), keepdims=True)
        elif self.mode == 'zero-one':
            if self.image_base:
                self.shift = image.min()
                self.scale = image.max() - self.shift
            else:
                self.shift = image.min(axis=(0, 1), keepdims=True)
                self.scale = image.max(axis=(0, 1), keepdims=True) - self.shift
        elif self.mode == 'negative-positive-one':
            if self.image_base:
                xmin, xmax = image.min(), image.max()
                self.shift = 0.5 * (xmax + xmin)
                self.scale = 0.5 * (xmax - xmin)
            else:
                xmin = image.min(axis=(0, 1), keepdims=True)
                xmax = image.max(axis=(0, 1), keepdims=True)
                self.shift = 0.5 * (xmax + xmin)
                self.scale = 0.5 * (xmax - xmin)
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
class CVBrightness(CVImageBase):
    """Adjust brightness of an image

    Args:
        brightness_factor (float | tuple[float, float] | list[float, float]):
            - float: How much to adjust the brightness.
                Can be any non-negative number. 0 gives a black image, 1 gives
                the original image while 2 increases the brightness by a factor of 2.
            - tuple[float, float]: Range of the ratio of brightness.
            - list[float, ... , float]: List of the ratio of brightness to be randomly chosen.
    """

    def __init__(self, brightness_factor):
        super().__init__()
        _input_check_value_range_set(brightness_factor)
        self.brightness_factor = brightness_factor

    def calculate(self, image):
        brightness_factor = _input_get_value_range_set(self.brightness_factor)
        image = _blend_cv(image, np.zeros_like(image), brightness_factor)
        return image

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(brightness_factor={self.brightness_factor})'
        return repr_str


@TRANSFORMS.register_module()
class CVRandomBrightness(CVImageBase):
    """Adjust brightness of an image

    Args:
        prob (float): Probability of image adjusting brightness.
        brightness_factor (float  | tuple[float, float] |  list[float, float]):
            - float : How much to adjust the brightness.
                Can be any non-negative number. 0 gives a black image, 1 gives
                the original image while 2 increases the brightness by a factor of 2.
            - tuple[float, float]: Range of the ratio of brightness .
            - list[float, ... , float] : list of the ratio of brightness to be randomly chosen.
    """

    def __init__(self, prob, brightness_factor=None):
        super().__init__()
        _input_check_value_range_set(brightness_factor)
        self.prob = prob
        self.brightness_factor = brightness_factor

    def calculate(self, image):
        if random.random() < self.prob:
            brightness_factor_sel = _input_get_value_range_set(self.brightness_factor)
            image = _blend_cv(image, np.zeros_like(image), brightness_factor_sel)
        return image

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'brightness_factor={self.brightness_factor})'
        return repr_str


def _contrast(img, contrast_factor):
    _assert_channels_cv(img, [3, 1])
    c = _get_dimensions_cv(img)[IMG_CHANNEL_AXIS]
    dtype = img.dtype if _is_floating_point_cv(img.dtype) else np.float32

    if c == 3:
        mean = np.mean(_rgb_to_grayscale_cv(img).astype(dtype), axis=(IMG_ROW_AXIS, IMG_COL_AXIS))
    else:
        mean = np.mean(img.astype(dtype), axis=(IMG_ROW_AXIS, IMG_COL_AXIS))

    return _blend_cv(img, mean, contrast_factor)


@TRANSFORMS.register_module()
class CVContrast(CVImageBase):
    """Adjust contrast of an image.

    Args:
        contrast_factor (float  | tuple[float, float] |  list[float, float] ):
            - float : How much to adjust the contrast. Can be any
                    non-negative number. 0 gives a solid gray image, 1 gives the
                    original image while 2 increases the contrast by a factor of 2.
            - tuple[float, float]:  Range of the ratio of contrast.
            - list[float, ... , float] : list of the ratio of contrast to be randomly chosen.
    """

    def __init__(self, contrast_factor):
        super().__init__()
        _input_check_value_range_set(contrast_factor)
        self.contrast_factor = contrast_factor

    def calculate(self, img):
        contrast_factor_sel = _input_get_value_range_set(self.contrast_factor)
        return _contrast(img, contrast_factor_sel)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(contrast_factor={self.contrast_factor})'
        return repr_str


@TRANSFORMS.register_module()
class CVRandomContrast(CVImageBase):
    """Adjust contrast of an image.

    Args:
        prob (float): Probability of image adjusting contrast.
        contrast_factor (float  | tuple[float, float] |  list[float, float] ):
            - float : How much to adjust the contrast. Can be any
                    non-negative number. 0 gives a solid gray image, 1 gives the
                    original image while 2 increases the contrast by a factor of 2.
            - tuple[float, float]:  Range of the ratio of contrast.
            - list[float, ... , float] : list of the ratio of contrast to be randomly chosen.
    """

    def __init__(self, prob, contrast_factor=None):
        super().__init__()
        _input_check_value_range_set(contrast_factor)
        self.prob = prob
        self.contrast_factor = contrast_factor

    def calculate(self, img):
        if random.random() < self.prob:
            contrast_factor_sel = _input_get_value_range_set(self.contrast_factor)
            img = _contrast(img, contrast_factor_sel)
        return img

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'contrast_factor={self.contrast_factor})'
        return repr_str


@TRANSFORMS.register_module()
class CVSaturation(CVImageBase):
    """Adjust color saturation of an image.

    Args:
        saturation_factor (float  | tuple[float, float] |  list[float, float] ):
            - float:  How much to adjust the saturation. 0 will
                    give a black and white image, 1 will give the original image while
                    2 will enhance the saturation by a factor of 2.
            - tuple[float, float]: Range of the ratio of saturation.
            - list[float, ... , float] : list of the ratio of saturation to be randomly chosen.
    """

    def __init__(self, saturation_factor):
        super().__init__()
        _input_check_value_range_set(saturation_factor)
        self.saturation_factor = saturation_factor

    def calculate(self, image):
        saturation_factor_sel = _input_get_value_range_set(self.saturation_factor)
        image = _blend_cv(image, _rgb_to_grayscale_cv(image), saturation_factor_sel)
        return image

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(saturation_factor={self.saturation_factor})'
        return repr_str


@TRANSFORMS.register_module()
class CVRandomSaturation(CVImageBase):
    """Adjust color saturation of an image.

    Args:
        prob (float):Probability of image adjusting sharpness.
        saturation_factor (float  | tuple[float, float] |  list[float, float] ):
            - float:  How much to adjust the saturation. 0 will
                    give a black and white image, 1 will give the original image while
                    2 will enhance the saturation by a factor of 2.
            - tuple[float, float]: Range of the ratio of saturation.
            - list[float, ... , float] : list of the ratio of saturation to be randomly chosen.
    """

    def __init__(self, prob, saturation_factor):
        super().__init__()
        _input_check_value_range_set(saturation_factor)
        self.prob = prob
        self.saturation_factor = saturation_factor

    def calculate(self, image):
        if random.random() < self.prob:
            saturation_factor_sel = _input_get_value_range_set(self.saturation_factor)
            image = _blend_cv(image, _rgb_to_grayscale_cv(image), saturation_factor_sel)
        return image

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'saturation_factor={self.saturation_factor})'
        return repr_str


def _rgb2hsv(img):
    r, g, b = img[..., 0], img[..., 1], img[..., 2]

    # Implementation is based on https://github.com/python-pillow/Pillow/blob/4174d4267616897df3746d315d5a2d0f82c656ee/
    # src/libImaging/Convert.c#L330
    maxc = np.max(img, axis=-1)
    minc = np.min(img, axis=-1)

    # The algorithm erases S and H channel where `maxc = minc`. This avoids NaN
    # from happening in the results, because
    #   + S channel has division by `maxc`, which is zero only if `maxc = minc`
    #   + H channel has division by `(maxc - minc)`.
    #
    # Instead of overwriting NaN afterwards, we just prevent it from occurring, so
    # we don't need to deal with it in case we save the NaN in a buffer in
    # backprop, if it is ever supported, but it doesn't hurt to do so.
    eqc = maxc == minc

    cr = maxc - minc
    # Since `eqc => cr = 0`, replacing denominator with 1 when `eqc` is fine.
    ones = np.ones_like(maxc)
    s = cr / np.where(eqc, ones, maxc)
    # Note that `eqc => maxc = minc = r = g = b`. So the following calculation
    # of `h` would reduce to `bc - gc + 2 + rc - bc + 4 + rc - bc = 6` so it
    # would not matter what values `rc`, `gc`, and `bc` have here, and thus
    # replacing denominator with 1 when `eqc` is fine.
    cr_divisor = np.where(eqc, ones, cr)
    rc = (maxc - r) / cr_divisor
    gc = (maxc - g) / cr_divisor
    bc = (maxc - b) / cr_divisor

    hr = (maxc == r) * (bc - gc)
    hg = ((maxc == g) & (maxc != r)) * (2.0 + rc - bc)
    hb = ((maxc != g) & (maxc != r)) * (4.0 + gc - rc)
    h = hr + hg + hb
    h = np.fmod((h / 6.0 + 1.0), 1.0)
    return np.concatenate((h[..., None], s[..., None], maxc[..., None]), axis=-1)


def _hsv2rgb(hsv_image):
    # 输入图像的形状：(height, width, 3)，通道顺序为(H, S, V)
    # 输出RGB图像，通道顺序为(R, G, B)

    h, s, v = hsv_image[..., 0], hsv_image[..., 1], hsv_image[..., 2]

    h_i = (h * 6).astype(int)
    f = h * 6 - h_i
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))

    r = np.where(h_i == 0, v,
                 np.where(h_i == 1, q, np.where(h_i == 2, p, np.where(h_i == 3, p, np.where(h_i == 4, t, v)))))
    g = np.where(h_i == 0, t,
                 np.where(h_i == 1, v, np.where(h_i == 2, v, np.where(h_i == 3, q, np.where(h_i == 4, p, p)))))
    b = np.where(h_i == 0, p,
                 np.where(h_i == 1, p, np.where(h_i == 2, t, np.where(h_i == 3, v, np.where(h_i == 4, v, q)))))
    rgb_image = np.stack((r, g, b), axis=-1)
    return rgb_image


def _hue(image, hue_factor):
    # hsv : 在颜色空间中,          h范围:0-360,s:范围0-100,v:范围0-100
    # opencv hsv : RGB2HSV_FULL :h范围:0-255,s:范围0-255,v:范围0-255
    #              RGB2HSV      :h范围:0-180,s:范围0-255,v:范围0-255

    #         hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV_FULL)
    #         h, s, v = cv2.split(hsv)
    #         h = h /255 *180## 注意opencv的h在COLOR_RGB2HSV_FULL范围时是255,
    #         h = ((h + 180* self.hue_factor % 180) / 180. * 255.).astype(hsv.dtype)
    #         image = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2RGB_FULL)
    ## 此处的方法和上面的方法等效
    #     hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)  ## 注意opencv的h在COLOR_RGB2HSV范围时是180
    #     h, s, v = cv2.split(hsv)
    #     h = (h + 180 * hue_factor % 180).astype(hsv.dtype)  ## 一个圈,所以是取余数
    #     image = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2RGB)
    ## 此处的方法和上面的方法等效
    image = _rgb2hsv(image)
    h, s, v = image[..., 0][..., None], image[..., 1][..., None], image[..., 2][..., None]
    h = (h + hue_factor) % 1.0
    image = np.concatenate((h, s, v), axis=-1)
    image = _hsv2rgb(image)

    return image


@TRANSFORMS.register_module()
class CVHue(CVImageBase):
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
        ## 注意,pytorch的hue范围是[-0.5, 0.5],这里和pytorch保持一致
        _input_check_value_range_set(hue_factor)
        self.hue_factor = hue_factor

    def calculate(self, image):
        hue_factor_sel = _input_get_value_range_set(self.hue_factor)
        image = _hue(image, hue_factor_sel)
        return image

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(hue_factor={self.hue_factor})'
        return repr_str


@TRANSFORMS.register_module()
class CVRandomHue(CVImageBase):
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
        ## 注意,pytorch的hue范围是[-0.5, 0.5],这里和pytorch保持一致
        _input_check_value_range_set(hue_factor)
        self.prob = prob
        self.hue_factor = hue_factor

    def calculate(self, image):
        if random.random() < self.prob:
            hue_factor_sel = _input_get_value_range_set(self.hue_factor)
            image = _hue(image, hue_factor_sel)
        return image

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'hue_factor={self.hue_factor})'
        return repr_str


@TRANSFORMS.register_module()
class CVSharpness(CVImageBase):
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

    def calculate(self, img):
        _assert_channels_cv(img, [1, 3])
        if img.shape[IMG_ROW_AXIS] <= 2 or img.shape[IMG_COL_AXIS] <= 2:
            return img
        sharpness_factor_sel = _input_get_value_range_set(self.sharpness_factor)
        return _blend_cv(img, _blurred_degenerate_image_cv(img), sharpness_factor_sel)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(sharpness_factor={self.sharpness_factor})'
        return repr_str


@TRANSFORMS.register_module()
class CVRandomSharpness(CVImageBase):
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

    def __init__(self, prob, sharpness_factor):
        super().__init__()
        _input_check_value_range_set(sharpness_factor)
        self.prob = prob
        self.sharpness_factor = sharpness_factor

    def calculate(self, img):
        if random.random() < self.prob:
            _assert_channels_cv(img, [1, 3])
            if img.shape[IMG_ROW_AXIS] <= 2 or img.shape[IMG_COL_AXIS] <= 2:
                return img
            sharpness_factor_sel = _input_get_value_range_set(self.sharpness_factor)
            img = _blend_cv(img, _blurred_degenerate_image_cv(img), sharpness_factor_sel)
        return img

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'sharpness_factor={self.sharpness_factor})'
        return repr_str


def _posterize(img, bits):
    if img.ndim < 3:
        raise TypeError(f"Input image tensor should have at least 3 dimensions, but found {img.ndim}")
    if img.dtype != np.uint8:
        raise TypeError(f"Only torch.uint8 image tensors are supported, but found {img.dtype}")

    _assert_channels_cv(img, [1, 3])
    mask = -int(2 ** (8 - bits))  # JIT-friendly for: ~(2 ** (8 - bits) - 1)
    return img & mask


@TRANSFORMS.register_module()
class CVPosterize(CVImageBase):
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

    def calculate(self, img):
        self.posterize_bins = _input_get_value_range_set(self.posterize_bins, 'int')
        if _is_floating_point_cv(img.dtype):
            img = (img * 255).astype(np.uint8)
            img = _posterize(img, self.posterize_bins)
            img = img.astype('float32') / 255.0
        else:
            img = _posterize(img, self.posterize_bins)
        return img

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(posterize_bins={self.posterize_bins})'
        return repr_str


@TRANSFORMS.register_module()
class CVRandomPosterize(CVImageBase):
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

    def calculate(self, img):
        if random.random() < self.prob:
            posterize_bins_sel = _input_get_value_range_set(self.posterize_bins, dtype='int')
            if _is_floating_point_cv(img.dtype):
                img = (img * 255).astype(np.uint8)
                img = _posterize(img, posterize_bins_sel)
                img = img.astype('float32') / 255.0
            else:
                img = _posterize(img, posterize_bins_sel)
        return img

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'posterize_bins={self.posterize_bins})'
        return repr_str


def _solarize(img, threshold):
    if img.ndim < 3:
        raise TypeError(f"Input image tensor should have at least 3 dimensions, but found {img.ndim}")

    _assert_channels_cv(img, [1, 3])

    if threshold > _max_value_cv(img.dtype):
        raise TypeError("Threshold should be less than bound of img.")

    inverted_img = _invert(img)
    return np.where(img >= threshold, inverted_img, img)


@TRANSFORMS.register_module()
class CVSolarize(CVImageBase):
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

    def calculate(self, img):
        solarize_threshold_sel = _input_get_value_range_set(self.solarize_threshold)
        if not _is_floating_point_cv(img.dtype):
            solarize_threshold_sel = solarize_threshold_sel * 255
        img = _solarize(img, solarize_threshold_sel)
        return img

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(solarize_threshold={self.solarize_threshold})'
        return repr_str


@TRANSFORMS.register_module()
class CVRandomSolarize(CVImageBase):
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

    def calculate(self, img):
        if random.random() < self.prob:
            solarize_threshold_sel = _input_get_value_range_set(self.solarize_threshold)
            if not _is_floating_point_cv(img.dtype):
                solarize_threshold_sel = solarize_threshold_sel * 255
            img = _solarize(img, solarize_threshold_sel)
        return img

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'solarize_threshold={self.solarize_threshold})'
        return repr_str


def _auto_contrast(img):
    ## img shape [row,col,channel]
    bound = _max_value_cv(img.dtype)
    dtype = img.dtype if _is_floating_point_cv(img.dtype) else np.float32
    minimum = np.amin(img, axis=(IMG_ROW_AXIS, IMG_COL_AXIS)).astype(
        dtype)
    maximum = np.amax(img, axis=(IMG_ROW_AXIS, IMG_COL_AXIS)).astype(dtype)
    scale = bound / (maximum - minimum)
    eq_idxs = np.logical_not(np.isfinite(scale))
    minimum[eq_idxs] = 0
    scale[eq_idxs] = 1
    return np.clip(((img - minimum) * scale), 0, bound).astype(img.dtype)


@TRANSFORMS.register_module()
class CVAutoContrast(CVImageBase):
    """Maximize contrast of an image by remapping its
    pixels per channel so that the lowest becomes black and the lightest
    becomes white.
    """

    def __init__(self):
        super().__init__()

    def calculate(self, img):
        return _auto_contrast(img)

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@TRANSFORMS.register_module()
class CVRandomAutoContrast(CVImageBase):
    """Maximize contrast of an image by remapping its
    pixels per channel so that the lowest becomes black and the lightest
    becomes white.

    Args:
        prob  (float):  Probability of image adjusting contrast.

    """

    def __init__(self, prob):
        super().__init__()
        self.prob = prob

    def calculate(self, img):
        if random.random() < self.prob:
            img = _auto_contrast(img)
        return img

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob})'
        return repr_str


def _scale_channel(img_chan):
    # TODO: we should expect bincount to always be faster than histc, but this
    # isn't always the case. Once
    # https://github.com/pytorch/pytorch/issues/53194 is fixed, remove the if
    # block and only use bincount.

    hist, bins = np.histogram(img_chan.reshape(-1), bins=256, range=(0, 255))

    nonzero_hist = hist[hist != 0]
    step = np.floor(nonzero_hist[:-1].sum() / 255.)
    if step == 0:
        return img_chan

    lut = np.floor((np.cumsum(hist, 0) + np.floor(step / 2)) / step)
    lut = np.clip(np.pad(lut, (1, 0), 'constant')[:-1], 0, 255)

    return lut[img_chan.astype(np.int64)]  # .astype(np.uint8)


def _equalize(img):
    c = _get_dimensions_cv(img)[IMG_CHANNEL_AXIS]
    if c == 3:
        # img_list = [cv2.equalizeHist(img[..., index])[..., None] for index in range(c)]
        img_list = [_scale_channel(img[..., index])[..., None] for index in range(c)]
        img = np.concatenate(img_list, axis=IMG_CHANNEL_AXIS)
    else:
        # img = cv2.equalizeHist(img)
        img = _scale_channel(img)

    return img


@TRANSFORMS.register_module()
class CVEqualize(CVImageBase):
    """Equalize the histogram of an image by applying
    a non-linear mapping to the input in order to create a uniform
    distribution of grayscale values in the output.
    """

    def __init__(self):
        super().__init__()

    def calculate(self, img):
        if _is_floating_point_cv(img.dtype):
            img = (img * 255).astype(np.uint8)
            img = _equalize(img)
            img = img.astype(np.float32) / 255.0
        else:
            img = _equalize(img)
        return img

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@TRANSFORMS.register_module()
class CVRandomEqualize(CVImageBase):
    """Equalize the histogram of an image by applying
    a non-linear mapping to the input in order to create a uniform
    distribution of grayscale values in the output.

    Args:
        prob  (float):  Probability of image adjusting equalize.
    """

    def __init__(self, prob):
        super().__init__()
        self.prob = prob

    def calculate(self, img):
        if random.random() < self.prob:
            if _is_floating_point_cv(img.dtype):
                img = (img * 255).astype(np.uint8)
                img = _equalize(img)
                img = img.astype(np.float32) / 255.0
            else:
                img = _equalize(img)
        return img

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}) '
        return repr_str


def _invert(img):
    if img.ndim < 3:
        raise TypeError(f"Input image tensor should have at least 3 dimensions, but found {img.ndim}")

    _assert_channels_cv(img, [1, 3])

    return _max_value_cv(img.dtype) - img


@TRANSFORMS.register_module()
class CVInvert(CVImageBase):
    """Invert the colors of an RGB/grayscale image.
    """

    def __init__(self):
        super().__init__()

    def calculate(self, image):
        image = _invert(image)
        return image

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@TRANSFORMS.register_module()
class CVRandomInvert(CVImageBase):
    """Invert the colors of an RGB/grayscale image.

    Args:
        prob (float):  Probability of image adjusting invert.
    """

    def __init__(self, prob):
        super().__init__()
        self.prob = prob

    def calculate(self, image):
        if random.random() < self.prob:
            image = _invert(image)
        return image

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob})'
        return repr_str


@TRANSFORMS.register_module()
class CVGaussianBlur(CVImageBase):
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

    def calculate(self, image):
        sigma_sel = _input_get_value_range_set(self.sigma)
        kernel_size_sel = _gaussian_kernel_size_get_value_range_set(self.kernel_size)
        image = cv2.GaussianBlur(image, (kernel_size_sel, kernel_size_sel), sigma_sel)
        if len(image.shape) == 2:
            image = image[..., None]
        return image

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(sigma={self.sigma}, '
        repr_str += f'kernel_size={self.kernel_size})'
        return repr_str


@TRANSFORMS.register_module()
class CVRandomGaussianBlur(CVImageBase):
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

    def calculate(self, image):
        if random.random() < self.prob:
            sigma_sel = _input_get_value_range_set(self.sigma)
            kernel_size_sel = _gaussian_kernel_size_get_value_range_set(self.kernel_size)
            image = cv2.GaussianBlur(image, (kernel_size_sel, kernel_size_sel), sigma_sel)
            if len(image.shape) == 2:
                image = image[..., None]
        return image

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'sigma={self.sigma}, '
        repr_str += f'kernel_size={self.kernel_size})'
        return repr_str


def _gaussian_noise(image, std, mean):
    dtype = image.dtype
    bound = 1.0 if _is_floating_point_cv(image.dtype) else 255.0
    if _is_floating_point_cv(image.dtype):
        std, mean = std, mean
    else:
        std = std * 255.0
        mean = mean * 255.0
    noise = np.random.randn(*image.shape) * std + mean
    image = np.clip((image + noise), 0, bound).astype(dtype)
    return image


@TRANSFORMS.register_module()
class CVGaussianNoise(CVImageBase):
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

    def calculate(self, image):
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
class CVRandomGaussianNoise(CVImageBase):
    """Performs Gaussian noise on the image by given std.

    Args:
        prob ( float):  Probability of image adjusting Gaussian noise.
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

    def calculate(self, image):
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
    # alpha = np.array([0.4638, 0.2112, 0.1410])
    alpha = np.array([random.normalvariate(0, alphastd, ) for i in range(3)])  # 差异来自alpha的生成,torch的随机数和random的随机数不一致

    rgb = np.sum(eigvec * alpha * eigval, axis=1)

    image = image + rgb

    return image


@TRANSFORMS.register_module()
class CVLighting(CVImageBase):
    """Lighting noise(AlexNet - style PCA - based noise)
    """

    def __init__(self, alphastd, eigval, eigvec):
        super().__init__()
        _input_check_value_range_set(alphastd)
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def calculate(self, image):
        alphastd_sel = _input_get_value_range_set(self.alphastd)
        if _is_floating_point_cv(image.dtype):
            image = _lighting(image, alphastd_sel, self.eigvec, self.eigval)  # 差异来自alpha的生成,torch的随机数和random的随机数不一致
        else:
            image = image.astype('float32') / 255.
            image = _lighting(image, alphastd_sel, self.eigvec, self.eigval)  # 差异来自alpha的生成,torch的随机数和random的随机数不一致
            image = np.clip(image * 255., 0, 255).astype('uint8')
        return image

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(alphastd={self.alphastd}, '
        repr_str += f'eigval={self.eigval}, '
        repr_str += f'eigvec={self.eigvec})'
        return repr_str


@TRANSFORMS.register_module()
class CVRandomLighting(CVImageBase):
    """Lighting noise(AlexNet - style PCA - based noise)

    Args:
        prob (float):  Probability of image adjusting Lighting noise.
    """

    def __init__(self, prob):
        super().__init__()
        self.prob = prob

    def calculate(self, image):
        if np.random.random() < self.prob:
            alphastd = np.random.uniform(0, 1)
            eigval = np.random.uniform(0, 1, (image.shape[IMG_CHANNEL_AXIS]))
            eigvec = np.random.uniform(0, 1, (image.shape[IMG_CHANNEL_AXIS], image.shape[IMG_CHANNEL_AXIS]))

            if _is_floating_point_cv(image.dtype):
                image = _lighting(image, alphastd, eigvec, eigval)  # 差异来自alpha的生成,torch的随机数和random的随机数不一致
            else:
                image = image.astype('float32') / 255.
                image = _lighting(image, alphastd, eigvec, eigval)  # 差异来自alpha的生成,torch的随机数和random的随机数不一致
                image = np.clip(image * 255., 0, 255).astype('uint8')
        return image

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob})'
        return repr_str


@TRANSFORMS.register_module()
class CVRandomBlur(CVImageBase):
    def __init__(self, prob=0.5, kernel_size=(5, 5)):
        super().__init__()
        self.prob = prob
        self.kernel_size = kernel_size

    def calculate(self, image):
        if random.random() < self.prob:
            image = cv2.blur(image, self.kernel_size)
        return image

