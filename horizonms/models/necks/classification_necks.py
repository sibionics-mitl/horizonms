import torch
from torch import nn
from torch import Tensor
import math
from typing import List
from ...builder import NECKS


__all__ = ("ClassificationPoolingNecks")


@NECKS.register_module()
class ClassificationPoolingNecks(nn.Module):
    r"""Neck for classification task which consists of pooling and flatten.

    Args:
        pool (str): the pooling type. It is 'avg' for average pooling or 'max' for max pooling.
    """
    def __init__(
        self,
        pool: str = 'avg'
    ) -> None:
        super().__init__()
        assert pool in ['avg', 'max']
        if pool == 'average':
            self.pool = nn.AdaptiveAvgPool2d(1)
        else:
            self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x: Tensor) -> Tensor:
        if isinstance(x, list):
            x = [self.pool(v) for v in x]
            x = [torch.flatten(v, 1) for v in x]
            x = tuple(x)
        else:
            x = self.pool(x)
            x = torch.flatten(x, 1)
        return x
