import torch
from torch import nn
from torch import Tensor
import math
from typing import List
from ...builder import HEADS


__all__ = ("ClassificationMultiHeads")


@HEADS.register_module()
class ClassificationMultiHeads(nn.Module):
    r"""Head for classification task.

    Args:
        input_dim (int): the dimension of input.
        dropout (float): the dropout ratio.
        num_softmax_classes_list (List[int]): the number of classes in each softmax head.
        num_sigmoid_classes_list (List[int]): the number of classes in each sigmoid head.
        priors (List[float]): the prior for initializing the last Conv of each head.
    """
    def __init__(
        self,
        input_dim: int,
        dropout: float = 0.3,
        num_softmax_classes_list: List = [1, 1, 11, 53],
        num_sigmoid_classes_list: List = [1, 1],        
        priors: List[float] = None,
    ) -> None:
        super().__init__()
        self.classifiers = nn.ModuleList()
        for num_classes in num_softmax_classes_list:
            classifier = nn.Sequential(
                nn.Dropout(p=dropout, inplace=False),
                nn.Linear(input_dim, num_classes),
                nn.Softmax(dim=1))
            self.classifiers.append(classifier)
        
        for num_classes in num_sigmoid_classes_list:
            classifier = nn.Sequential(
                nn.Dropout(p=dropout, inplace=False),
                nn.Linear(input_dim, num_classes),
                nn.Sigmoid())
            self.classifiers.append(classifier)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

        self.priors = priors
        if self.priors is not None:
            if len(self.priors) == 1:
                for classifier in self.classifiers:
                    classifier[-1].bias.data.fill_(self.priors[0])
            else:
                with torch.no_grad():
                    for classifier in self.classifiers:
                        classifier[-1].bias.data = torch.as_tensor(self.priors, device=classifier[-1].bias.device)

    def forward(self, x: Tensor) -> Tensor:
        x = [classifier(x) for classifier in self.classifiers]
        return tuple(x)
