from typing import List, Tuple, Dict
import copy
import torch
from torch import Tensor
from .image import *
from .spatial import *
from .image_cv import *
from .spatial_cv import *
from .batch_transforms import *
from .base import Compose
from ..builder import TRANSFORMS


__all__ = ("CustomizedTrivialAugment", "HorizonmsTrivialAugment", "SequentialAugment")


SUPPORT_OPERATORS = [
    # image.py
    "Uint8ToFloat", "Identity",
    "Brightness", "Contrast", "Saturation", "Hue", "Sharpness",
    "Posterize", "Solarize", "AutoContrast", "Equalize", "Invert",
    "GaussianBlur", "GaussianNoise", "Lighting",
    # image_cv.py
    "CVUint8ToFloat", "CVIdentity",
    "CVBrightness", "CVContrast", "CVSaturation", "CVHue", "CVSharpness",
    "CVPosterize", "CVSolarize", "CVAutoContrast", "CVEqualize", "CVInvert", 
    "CVGaussianBlur", "CVGaussianNoise", "CVLighting", 
    # spatial
    "ShearX", "ShearY", "TranslateX", "TranslateY", 
    "CropX", "CropY", "Fliplr", "Flipud", "Rotate", 
    "Resize", "ResizeWidth",
    "ImagePadding", "ImageHeightPaddingOrCrop",
    "Scale",
    # spatial_cv
    "CVShearX", "CVShearY", "CVTranslateX", "CVTranslateY", 
    "CVCropX", "CVCropY", "CVFliplr", "CVFlipud", "CVRotate", 
    "CVResize", "CVResizeWidth", 
    "CVImagePadding",
    "CVScale",
]

SUPPORT_OPERATORS_HMS = [
    # image.py
    "Normalizer", 
    # image_cv.py
    "CVCvtColor", "CVNormalizer"
]

SUPPORT_OPERATORS_EXTRA = [
    # image.py
    "RandomBrightness", "RandomContrast", "RandomSaturation", "RandomHue", "RandomSharpness",
    "RandomPosterize", "RandomSolarize", "RandomAutoContrast", "RandomEqualize", "RandomInvert",
    "RandomGaussianBlur", "RandomGaussianNoise", "RandomLighting",
    # image_cv.py
    "CVRandomBrightness", "CVRandomContrast", "CVRandomSaturation", "CVRandomHue", "CVRandomSharpness",
    "CVRandomPosterize", "CVRandomSolarize", "CVRandomAutoContrast", "CVRandomEqualize", "CVRandomInvert",
    "CVRandomGaussianBlur", "CVRandomGaussianNoise", "CVRandomLighting", 
    # spatial
    "RandomResizedCrop", "RandomCrop",        
    "RandomShearX", "RandomShearY", "RandomTranslateX", "RandomTranslateY",
    "RandomCropX", "RandomCropY", "RandomFliplr", "RandomFlipud", "RandomRotate",
    "RandomScale",
    # spatial_cv
    "CVRandomResizedCrop", "CVRandomCrop", 
    "CVRandomScale", "CVRandomShift",
    "CVRandomShearX", "CVRandomShearY", "CVRandomTranslateX", "CVRandomTranslateY", 
    "CVRandomCropX", "CVRandomCropY", "CVRandomFliplr", "CVRandomFlipud", "CVRandomRotate",
    # other augment
    "CustomizedTrivialAugment", "HorizonmsTrivialAugment",
    # batch transforms
    "ToOnehotLabels", "Mixup", "SoftmaxLabelSmoothing", "SigmoidLabelSmoothing",
]


@TRANSFORMS.register_module()
class CustomizedTrivialAugment(torch.nn.Module):
    r"""Dataset-independent data-augmentation with TrivialAugment using customized augmentation operators, as described in
    `"TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation" <https://arxiv.org/abs/2103.10158>`.
    If the image is torch Tensor, it should be of type `torch.uint8`, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        augment_operators (List[dict]): augmentation operators and their parameters.
        num_magnitude_bins (int): number of magnitude.
    """

    def __init__(
            self,
            augment_operators: List[dict],
            num_magnitude_bins: int = 31,
    ) -> None:
        super().__init__()
        self.augment_operators = augment_operators
        self.num_magnitude_bins = num_magnitude_bins
        self._suppert_operators = set(SUPPORT_OPERATORS)
        operators = [aug['name'] for aug in augment_operators]

        assert set(operators).issubset(self._suppert_operators), f"Supported operators are {self._suppert_operators}"

    def forward(self, image: Tensor, target: Dict = None) -> Tuple[Tensor, Dict]:
        op_index = int(torch.randint(len(self.augment_operators), (1,)).item())
        op_setting = copy.deepcopy(self.augment_operators[op_index])
        op_name = op_setting.pop('name')
        if len(op_setting) == 0:
            op = eval(op_name)()
        else:
            op_param_range = op_setting.pop('param_range')
            magnitudes = torch.linspace(op_param_range[0], op_param_range[1], self.num_magnitude_bins)
            magnitude = (
                float(magnitudes[torch.randint(len(magnitudes), (1,), dtype=torch.long)].item())
                if magnitudes.ndim > 0
                else 0.0
            )
            # print(f"op name = {op_name}, magnitude = {magnitude}")
            op = eval(op_name)(magnitude, **op_setting)
        return op(image, target)

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += ", augment_operators={augment_operators}"
        s += ", num_magnitude_bins={num_magnitude_bins}"
        s += ")"
        return s.format(**self.__dict__)


@TRANSFORMS.register_module()
class HorizonmsTrivialAugment(torch.nn.Module):
    r"""Dataset-independent data-augmentation with modified TrivialAugment using customized augmentation operators, as described in
    `"TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation" <https://arxiv.org/abs/2103.10158>`.
    If the image is torch Tensor, it should be of type `torch.uint8`, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    The modification is done such that 1) it supports more than one random number for any operator, and
    2) the random number is not divided into bins.
    
    Args:
        augment_operators (List[dict]): augmentation operators and their parameters.
    """
    def __init__(
            self,
            augment_operators: List[dict],
    ) -> None:
        super().__init__()
        self.augment_operators = augment_operators
        self._suppert_operators = set(SUPPORT_OPERATORS + SUPPORT_OPERATORS_HMS)
        operators = [aug['name'] for aug in augment_operators]
        assert set(operators).issubset(self._suppert_operators), f"Supported operators are {self._suppert_operators}"

    def forward(self, image: Tensor, target: Dict = None) -> Tuple[Tensor, Dict]:
        op_index = int(torch.randint(len(self.augment_operators), (1,)).item())
        op_setting = copy.deepcopy(self.augment_operators[op_index])
        op_name = op_setting.pop('name')
        if len(op_setting) > 0:
            op = eval(op_name)(**op_setting)
        else:
            op = eval(op_name)()
        return op(image, target)

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "augment_operators={augment_operators}"
        s += ")"
        return s.format(**self.__dict__)


@TRANSFORMS.register_module()
class SequentialAugment(torch.nn.Module):
    r"""Sequential augmentation using customized augmentation operators.
    If the image is torch Tensor, it should be of type `torch.uint8`, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        augment_operators (List[dict]): augmentation operators and their parameters.
    """
    def __init__(
            self,
            augment_operators: List[dict],
    ) -> None:
        super().__init__()
        self.augment_operators = augment_operators
        self._suppert_operators = set(SUPPORT_OPERATORS + SUPPORT_OPERATORS_HMS + SUPPORT_OPERATORS_EXTRA)
        operators = [aug['name'] for aug in augment_operators]
        assert set(operators).issubset(self._suppert_operators), f"Supported operators are {self._suppert_operators}"
        
        transforms = []
        for op_setting_org in self.augment_operators:
            op_setting = copy.deepcopy(op_setting_org)
            op_name = op_setting.pop('name')
            if len(op_setting) > 0:
                op = eval(op_name)(**op_setting)
            else:
                op = eval(op_name)()
            transforms.append(op)
        self.transforms = Compose(transforms)

    def forward(self, image: Tensor, target: Dict = None) -> Tuple[Tensor, Dict]:
        return self.transforms(image, target)

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "augment_operators={augment_operators}"
        s += ")"
        return s.format(**self.__dict__)
