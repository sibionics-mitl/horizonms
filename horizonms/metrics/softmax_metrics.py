import torch
from torch import nn
from .softmax_metrics_func import softmax_accuracy, softmax_accuracy_topk, \
                softmax_cohen_kappa_score
from .. import METRICS


__all__ = ("SoftmaxAccuracy", "SoftmaxAccuracyTopk", "SoftmaxCohenKappaScore")


@METRICS.register_module()
class SoftmaxAccuracy():
    r"""Accuracy for softmax output.
    """
    def __call__(self, ytrue, ypred):
        return softmax_accuracy(ytrue, ypred)


@METRICS.register_module()
class SoftmaxAccuracyTopk():
    r"""Top-k accuracy for softmax output.

    Args:
        k (int): parameter in top-k.
    """
    def __init__(self, k: int):
        self.k = k
        
    def __call__(self, ytrue, ypred):
        return softmax_accuracy_topk(ytrue, ypred, self.k)


@METRICS.register_module()
class SoftmaxCohenKappaScore():
    r"""Cohen's kappa for softmax output.

    Args:
        weights (str): the type of Cohen's kappa.
        category (bool): `category=True` converts both predictions and ground truths as one-hot. 
        epsilon (float): a small number for the stability of metric.
    """
    def __init__(self, weights: str = None, category: bool = True, epsilon: float = 1e-10):
        self.weights = weights
        self.category = category
        self.epsilon = epsilon

    def __call__(self, ytrue, ypred):
        kappa = softmax_cohen_kappa_score(ytrue, ypred, weights=self.weights,
                category=self.category, epsilon=self.epsilon)
        return kappa


