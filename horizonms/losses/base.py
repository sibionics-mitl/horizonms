import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod


__all__ = ("SoftmaxBaseLoss", "SigmoidBaseLoss")


class SoftmaxBaseLoss(ABC):
    r"""Base class for softmax loss.

    Args:
        missing_values (bool): whether to handle missing values.
        epsilon (float): a small number for the stability of loss.
    """
    def __init__(self, missing_values=False, epsilon=1e-6):
        self.missing_values = missing_values
        self.epsilon = epsilon

    def __call__(self, ytrue, ypred, *argv, **kwargs):
        ytrue, ypred = self.preprocess(ytrue, ypred)
        if self.missing_values:
            ytrue, ypred = self.remove_missing_values(ytrue, ypred)

        loss = self.calculate_loss(ytrue, ypred, *argv, **kwargs)

        return loss

    def preprocess(self, ytrue, ypred):
        num_classes = ypred.shape[1]
        if ytrue.dim() == ypred.dim():
            if ypred.dim() == 4:
                ytrue = ytrue.permute(0, 2, 3, 1).reshape(-1, num_classes)
                ypred = ypred.permute(0, 2, 3, 1).reshape(-1, num_classes)
        elif ytrue.dim() == ypred.dim() - 1:
            if ypred.dim() == 4:
                ytrue = ytrue.reshape(-1)
                ypred = ypred.permute(0, 2, 3, 1).reshape(-1, num_classes)
            ytrue = F.one_hot(ytrue.long(), num_classes).float()
        return ytrue, ypred

    def remove_missing_values(self, ytrue, ypred):
        flag = torch.isnan(ytrue).sum(axis=1) == 0
        ytrue = ytrue[flag, :]
        ypred = ypred[flag, :]
        return ytrue, ypred

    @abstractmethod
    def calculate_loss(self, ytrue, ypred, **kwargs):
        pass


class SigmoidBaseLoss(ABC):
    r"""Base class for sigmoid loss.

    Args:
        missing_values (bool): whether to handle missing values.
        epsilon (float): a small number for the stability of loss.
    """
    def __init__(self, missing_values=False, epsilon=1e-6):
        self.missing_values = missing_values
        self.epsilon = epsilon

    def __call__(self, ytrue, ypred, *argv, **kwargs):
        ytrue, ypred = self.preprocess(ytrue, ypred)
        if self.missing_values:
            flag = self.detect_existing_values(ytrue)
        else:
            flag = None

        if flag is None:
            loss = self.calculate_loss(ytrue, ypred, *argv, **kwargs)
        else:
            loss = self.calculate_loss(ytrue, ypred, flag, *argv, **kwargs)

        return loss

    def preprocess(self, ytrue, ypred):
        num_classes = ytrue.shape[1]
        if ytrue.dim() == 4:
            ytrue = ytrue.permute(0, 2, 3, 1).reshape(-1, num_classes)
            ypred = ypred.permute(0, 2, 3, 1).reshape(-1, num_classes)
        return ytrue, ypred

    def detect_existing_values(self, ytrue):
        flag = torch.isnan(ytrue).logical_not_()
        return flag

    @abstractmethod
    def calculate_loss(self, ytrue, ypred, **kwargs):
        pass
