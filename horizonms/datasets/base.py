from typing import Any, Tuple, Dict
from abc import ABC, abstractmethod
import numpy as np
import torch
from torch.utils.data import Dataset
from .. import transforms as T
from ..builder import build_transforms

__all__ = ("BaseDataset")


class BaseDataset(Dataset, ABC):
    r"""Base class for Dataset.

    Args:
        transforms_pt (callable, optional): a data augmentation object that is implemented by PyTorch.
        transforms_cv (callable, optional): a data augmentation object that is implemented by OpenCV.
        to_tensor (bool): if True, converts the samples into PyTorch Tensor.
    """
    def __init__(self, transforms_pt=None, transforms_cv=None, to_tensor=True):
        super(BaseDataset, self).__init__()
        if isinstance(transforms_pt, dict):
            self.transforms_pt = build_transforms(transforms_pt)
        else:
            self.transforms_pt = transforms_pt
        if (self.transforms_pt is not None) & isinstance(self.transforms_pt, list):   
            self.transforms_pt = T.Compose(self.transforms_pt)
        if isinstance(transforms_cv, dict):
            self.transforms_cv = build_transforms(transforms_cv)
        else:
            self.transforms_cv = transforms_cv
        if (self.transforms_cv is not None) & isinstance(self.transforms_cv, list):
            self.transforms_cv = T.Compose(self.transforms_cv)
        self.to_tensor = to_tensor

    def __len__(self) -> int:
        r"""get the number of samples in the dataset.
        """
        return len(self.get_images())

    @abstractmethod
    def get_images(self):
        r"""gets image names in the dataset.
        """
        pass

    @property
    def len(self) -> int:
        r"""the number of samples in the dataset.
        """
        return len(self.get_images())

    @abstractmethod
    def getitem(self, index: int) -> Tuple[Any, Any]:
        r"""gets image and target for a single sample.

        Args:
            index (int): index of the sample in the dataset.
        """
        pass

    def get_target_single_item(self, key: str, value: Any, type: str = None):
        r"""formats single item in target as a predefined dictionary.
        """
        return {key: dict(type=type, value=value)}

    def format_target(self, target: Dict[str, Dict[str, Any]]) -> Dict[str, T.TargetStructure]:
        r"""formats target by `T.TargetStructure`.
        """
        target_format = dict()
        for key, value in target.items():
            target_format[key] = T.TargetStructure(**value)
        return target_format

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        r"""gets image and target for a single sample, achieves data augmentation, and converts them into PyTorch Tensor if `self.to_tensor=True`.

        Args:
            index (int): index of the sample in the dataset.
        """
        image, target = self.getitem(index)
        target = self.format_target(target)

        if self.transforms_cv is not None:
            image, target = self.transforms_cv(image, target)

        if self.to_tensor:
            if image.ndim == 2:
                image = image[:, :, None]
            image = torch.from_numpy(image.transpose((2, 0, 1)).copy()).contiguous()
            if target is not None:
                for key, value in target.items():
                    if value.type == "masks":
                        if value.value.ndim == 2:
                            value.value = value.value[:, :, None]
                        if value.islist:
                            value.value = [np.ascontiguousarray(v.transpose(2, 0, 1)) for v in value.value]
                        else:
                            value.value = np.ascontiguousarray(value.value.transpose(2, 0, 1))
                    if value.islist:
                        check_str = [type(v) == str for v in value.value]
                        if sum(check_str) > 0:
                            value = T.TargetStructure(type=value.type, value=value.value)
                        else:
                            value = T.TargetStructure(type=value.type,
                                                      value=[torch.tensor(v).float() for v in value.value])
                    else:
                        if type(value.value) == str:
                            value = T.TargetStructure(type=value.type, value=value.value)
                        else:
                            value = T.TargetStructure(type=value.type, value=torch.tensor(value.value).float())
                    target[key] = value

        if self.transforms_pt is not None:
            image, target = self.transforms_pt(image, target)

        return image, target
