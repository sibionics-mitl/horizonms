from tkinter import W
import torch
from torch import Tensor
from typing import Tuple
import numpy as np
import torch.nn.functional as F
from ..builder import TRANSFORMS


__all__ = ("ToOnehotLabels", "Mixup", "SoftmaxLabelSmoothing", "SigmoidLabelSmoothing")


@TRANSFORMS.register_module()
class ToOnehotLabels(torch.nn.Module):
    r"""Convert labels into one-hot encoding.

    Args:
        num_classes (int): number of classes.
    """
    def __init__(self, num_classes, index=None):
        super().__init__()
        self.num_classes = num_classes
        self.index = index

    def forward(self, images, targets):
        for key in targets.keys():
            if targets[key].type == 'labels':
                value = targets[key].value
                if self.index is None:
                    value = F.one_hot(value.long(), self.num_classes).float()
                else:
                    for idx in self.index:
                        value[idx] = F.one_hot(value[idx].long(), self.num_classes).float()
                targets[key].value = value
        return images, targets


@TRANSFORMS.register_module()
class SoftmaxLabelSmoothing(torch.nn.Module):
    r"""Achieve label smoothing for multi-class classification task.

    Args:
        smoothing (float): smoothing value.
    """
    def __init__(self, smoothing=0.1, index=None):
        super().__init__()
        if isinstance(smoothing, list):
            assert len(smoothing) == len(index), "The length of smoothing and that of index have to be same."
            assert all( [0 < s < 1.0 for s in smoothing]), "All elements in smoothing should be in [0,1]."
        else:
            assert 0 < smoothing < 1.0, "The value of smoothing should be in [0,1]."
        self.smoothing = smoothing
        self.index = index
        if self.index is not None:
            if isinstance(smoothing, float):
                self.smoothing = [smoothing] * len(self.index)

    def forward(self, images, targets):
        for key in targets.keys():
            if targets[key].type == 'labels':
                value = targets[key].value
                if self.index is None:
                    value = value * (1 - self.smoothing) + \
                            (1 - value) * self.smoothing / (value.shape[1] - 1)
                else:
                    for idx, s in zip(self.index, self.smoothing):
                        value[idx] = value[idx] * (1 - s) + \
                                (1 - value[idx]) * s / (value[idx].shape[1] - 1)
                targets[key].value = value
        return images, targets


@TRANSFORMS.register_module()
class SigmoidLabelSmoothing(torch.nn.Module):
    r"""Achieve label smoothing for multi-label classification task.

    Args:
        smoothing (float): smoothing value.
    """
    def __init__(self, smoothing=0.1, index=None):
        assert 0 < smoothing < 1.0
        if isinstance(smoothing, list):
            len(smoothing) == len(index)
            assert all( [0 < s < 1.0 for s in smoothing]), "All elements in smoothing should be in [0,1]."
        else:
            assert 0 < smoothing < 1.0, "The value of smoothing should be in [0,1]."
        self.smoothing = smoothing
        self.index = index
        if self.index is not None:
            if isinstance(smoothing, float):
                self.smoothing = [smoothing] * len(self.index)

    def __call__(self, images, targets):
        for key in targets.keys():
            if targets[key].type == 'labels':
                value = targets[key].value
                if self.index is None:
                    value = value * (1 - self.smoothing) + (1 - value) * self.smoothing
                else:
                    for idx, s in zip(self.index, self.smoothing):
                        value[idx] = value[idx] * (1 - s) + (1 - value[idx]) * s
                targets[key].value = value
        return images, targets


@TRANSFORMS.register_module()
class Mixup(torch.nn.Module):
    r"""Mixup data augmentation operator.

    Args:
        alpha (float): mixup parameter.
    """
    def __init__(self, alpha=0.2, index=None):
        super().__init__()
        self.alpha = alpha
        self.index = index
        assert index is None, "Mixup is only supported for one classification head."

    def forward(self, images, targets) -> Tuple[Tensor, dict]:
        batch_size = images.shape[0]
        lam = np.random.beta(self.alpha, self.alpha)
        indices = torch.randperm(batch_size, device=images.device, dtype=torch.long)
        images = lam * images + (1 - lam) * images[indices, :]
        for key in targets.keys():
            if targets[key].type == 'labels':
                targets[key].value = lam * targets[key].value + (1 - lam) * targets[key].value[indices]
        return images, targets
