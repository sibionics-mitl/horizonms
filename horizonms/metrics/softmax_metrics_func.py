import torch
from torch import nn


__all__ = ("softmax_accuracy", "softmax_accuracy_topk", "softmax_cohen_kappa_score")


def softmax_accuracy(ytrue, ypred):
    r"""Accuracy for softmax output.

    Args:
        ytrue (Tensor): ground truth.
        ypred (Tensor): prediction.
    """
    assert (ytrue.dim() in [1,2]) & (ypred.dim() == 2)
    if ytrue.dim() == 2:
        ytrue = torch.argmax(ytrue, dim=1)
    ypred = torch.argmax(ypred, dim=1)
    acc = (ytrue == ypred).sum() / float(len(ytrue))
    return acc


def softmax_accuracy_topk(ytrue, ypred, topk=1):
    r"""Top-k accuracy for softmax output.

    Args:
        ytrue (Tensor): ground truth.
        ypred (Tensor): prediction.
        k (int): parameter in top-k.
    """
    assert (ytrue.dim() in [1,2]) & (ypred.dim() == 2)
    if ytrue.dim() == 2:
        ytrue = torch.argmax(ytrue, dim=1)
    _, ypred = torch.topk(ypred, k=topk, dim=1, largest=True, sorted=True)
    acc = (ytrue[:, None] == ypred).sum() / float(len(ytrue))
    return acc


def softmax_cohen_kappa_score(ytrue, ypred, weights=None, category=True, epsilon=1e-10):
    r"""Cohen's kappa for softmax output.

    Args:
        ytrue (Tensor): ground truth.
        ypred (Tensor): prediction.
        weights (str): the type of Cohen's kappa.
        category (bool): `category=True` converts both predictions and ground truths as one-hot. 
        epsilon (float): a small number for the stability of metric.
    """
    assert ytrue.dim() == 2
    nb_samples, nb_classes = ytrue.shape
    if category:
        device = ytrue.device
        ypred = torch.eye(nb_classes, device=device)[torch.argmax(ypred, dim=1)]
        ytrue = torch.eye(nb_classes, device=device)[torch.argmax(ytrue, dim=1)]
    
    if weights is None:
        weights = torch.ones((nb_classes, nb_classes)) - torch.eye(nb_classes)
    elif weights == 'linear':
        x = torch.arange(nb_classes).reshape(-1, 1)
        weights = torch.abs(x - x.T) / (nb_classes - 1)
    elif weights == 'quadratic':
        x = torch.arange(nb_classes).reshape(-1, 1)
        weights = (x - x.T)**2 / (nb_classes - 1)**2
    weights = torch.tensor(weights, dtype=ytrue.dtype, device=ytrue.device)
    assert weights.shape == (nb_classes, nb_classes)

    hist_true = ytrue.sum(dim=0)
    hist_pred = ypred.sum(dim=0) + epsilon
    conf_mat = torch.matmul(ypred.T, ytrue)
    conf_expect = torch.matmul(torch.reshape(hist_pred, (-1, 1)), torch.reshape(hist_true, (1, -1)))
    
    nom = (weights * conf_mat).sum()
    denom = (weights * conf_expect).sum() / nb_samples
    kappa = 1 - nom / denom

    return kappa