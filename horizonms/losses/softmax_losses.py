import torch
from ..metrics import softmax_cohen_kappa_score
from .base import SoftmaxBaseLoss
from .. import LOSSES


__all__ = ("SoftmaxCohenKappaLoss", "SoftmaxFocalLoss", "SoftmaxMixFocalLoss", "SoftmaxCrossEntropyLoss")


@LOSSES.register_module()
class SoftmaxCohenKappaLoss(SoftmaxBaseLoss):
    r"""Cohen's kappa loss for softmax output.

    Args:
        weights (str): the type of Cohen's kappa.
    """
    def __init__(self, weights="quadratic", *argv, **kwargs):
        super(SoftmaxCohenKappaLoss, self).__init__(*argv, **kwargs)
        self.weights = weights

    def calculate_loss(self, ytrue, ypred):
        kappa = softmax_cohen_kappa_score(ytrue, ypred, weights=self.weights, category=False)
        return 1 - kappa


@LOSSES.register_module()
class SoftmaxFocalLoss(SoftmaxBaseLoss):
    r"""Focal loss for softmax output.

    Args:
        mode (str): the mode of focal loss. It is 'all' or 'foreground'. Default: ``'all'``.
            These two modes are different in how to get focal loss from individual samples.
            'all' uses the average loss of all samples.
            'foreground' is similar as the original focal loss which only considers the foreground in the denominator.
        gamma (float): Exponent of the modulating factor to balance easy vs hard examples. Default: ``2.0``.
    """
    def __init__(self, mode='all', gamma=2.0, *argv, **kwargs):
        super(SoftmaxFocalLoss, self).__init__(*argv, **kwargs)
        self.mode = mode
        self.gamma = gamma

    def calculate_loss(self, ytrue, ypred, weights=None):
        ypred = torch.clamp(ypred, self.epsilon, 1-self.epsilon)
        losses = -ytrue*(1-ypred).pow(self.gamma)*torch.log(ypred)
        if weights is not None:
            losses = weights * losses
        losses = torch.sum(losses, dim=1)

        if self.mode=='all':
            loss  = losses.mean()
        elif self.mode=='foreground':
            # class 0 is background class, the others are foreground classes
            normalizer = ytrue[:, 1:].sum()
            normalizer = torch.max(torch.tensor(1, device=normalizer.device), normalizer)
            loss  = losses.sum()/normalizer
        return loss


@LOSSES.register_module()
class SoftmaxMixFocalLoss(SoftmaxBaseLoss):
    def __init__(self, mode='all', gamma=2.0, *argv, **kwargs):
        super(SoftmaxMixFocalLoss, self).__init__(*argv, **kwargs)
        self.mode = mode
        self.gamma = gamma

    def calculate_loss(self, ytrue, ypred, weights=None):
        ypred = torch.clamp(ypred, self.epsilon, 1-self.epsilon)
        losses = -ytrue * torch.log(ypred)  # CE on foreground class
        losses_bkg = losses[:, 0] * (1 - ypred[:, 0]).pow(self.gamma) # FL on background class
        if weights is not None:
            losses = weights * losses
            losses_bkg = weights[0] * losses_bkg 
        losses = torch.sum(losses[:, 1:], dim=1) + losses_bkg

        if self.mode=='all':
            loss  = losses.mean()
        elif self.mode=='foreground':
            # class 0 is background class, the others are foreground classes
            normalizer = ytrue[:, 1:].sum()
            normalizer = torch.max(torch.tensor(1, device=normalizer.device), normalizer)
            loss  = losses.sum()/normalizer
        return loss


@LOSSES.register_module()
class SoftmaxCrossEntropyLoss(SoftmaxBaseLoss):
    r"""Cross entropy loss for softmax output.

    Args:
        mode (str): the mode of cross entropy loss. It is 'all' or 'balance'. Default: ``'all'``.
            These two modes are different in how to get focal loss from individual samples.
            'all' returns an average for all samples.
            'balance' returns an average for each class.
    """
    def __init__(self, mode='all', *argv, **kwargs):
        super(SoftmaxCrossEntropyLoss, self).__init__(*argv, **kwargs)
        self.mode = mode

    def calculate_loss(self, ytrue, ypred):
        ypred = torch.clamp(ypred, self.epsilon, 1-self.epsilon)
        losses = -ytrue * torch.log(ypred)
        if self.mode=='all':
            loss = losses.sum(dim=1).mean()
        elif self.mode=='balance':
            loss = torch.sum(losses, dim=0)/(torch.sum(ytrue, dim=0) + self.epsilon)
        return loss
