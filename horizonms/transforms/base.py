import torch
from torch import Tensor
import numpy as np
from typing import Tuple, Any, Union, List


__all__ = ("TargetStructure", "Compose", "ToTensor")


class TargetStructure(object):
    r"""Definition of target structure.

    Args:
        type (str): type of target. It is `'bboxes'`, `'points'`, `'masks'`, `'labels'`, or `None`.
        value (Union[Any, List[Any]]): value of target.
    """
    def __init__(self, type: str=None, value: Union[Any, List[Any]]=None):
        assert type in ['bboxes', 'points', 'masks', 'labels', None]
        self.type = type
        self.value = value
        self.islist = False
        if isinstance(value, list):
            self.islist = True

    def to(self, device, *argv, **kwargs):
        if self.islist:
            cast_value = [v.to(device, *argv, **kwargs) for v in self.value]
        else:
            cast_value = self.value.to(device, *argv, **kwargs)            
        return TargetStructure(self.type, cast_value)

    def __repr__(self):
        d = dict(self.__dict__)
        if self.islist:
            shape =  f"list of {len(d['value'])} items"
            s = f'{self.__class__.__name__}(type={d["type"]}, value={shape})'
        else:
            if np.isscalar(d["value"]):
                s = f'{self.__class__.__name__}(type={d["type"]}, value={d["value"]})'
            # elif isinstance(d["value"], np.ndarray]) & (d["value"].size==1):
            #     s = f'{self.__class__.__name__}(type={d["type"]}, value={d["value"]})'
            else:
                shape =  d["value"].shape
                s = f'{self.__class__.__name__}(type={d["type"]}, value.shape={shape})'
        return s


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        if target is None:
            for t in self.transforms:
                # print(t, image.dtype, image.min(), image.max())
                image = t(image)
                # print(t, image.dtype, image.min(), image.max())
            return image
        else:
            for t in self.transforms:
                # print(">>>before", t, target, image.shape)
                image, target = t(image, target)
                # print(">>>after", t, target, image.shape)
            return image, target


class ToTensor(object):
    r"""Converts image and/or target into Pytorch Tensor.

    Args:
        dtype (str): type of image. It is `'uint8'` or `'float'`.
    """
    def __init__(self, dtype='float'):
        assert dtype in ['uint8', 'float']
        self.dtype = dtype

    def __call__(self, image, target=None) -> Tuple[Tensor, dict]:
        image = torch.from_numpy(image.transpose((2, 0, 1))).contiguous()
        if self.dtype == 'uint8':
            image = image.type(torch.uint8)
        else:
            image = image.float() / 255.0
        if target is None:
            return image
        else:
            return image, target
